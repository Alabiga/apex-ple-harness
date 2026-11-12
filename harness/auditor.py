"""Deterministic deliverable auditor + G=1 checkpoint manager.

Everything runs *inside the task container* through ``protocol.exec_command`` /
``write_file`` so it audits exactly the bytes the grader will see.
"""
from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import shlex
from pathlib import Path
from typing import Any, Iterable

from .protocol import PLEProtocol

log = logging.getLogger("apex.auditor")

CHECKPOINT_DIR = "/tmp/.apex_ple_checkpoint"
AUDIT_SCRIPT_PATH = "/tmp/.apex_audit.py"
AUDIT_ARGS_PATH = "/tmp/.apex_audit_args.json"
CHECKPOINT_SCRIPT_PATH = "/tmp/.apex_checkpoint.py"
CHECKPOINT_ARGS_PATH = "/tmp/.apex_checkpoint_args.json"

DELIVERABLE_EXTS = (
    "json", "yaml", "yml", "csv", "xlsx", "xls", "dxf", "ifc", "md", "txt", "py",
    "pdf", "docx", "png", "svg", "ipynb", "tsv", "xml", "dwg", "rvt", "docx",
)
_PATH_RE = re.compile(
    r"(?<![\w/.\\-])((?:/[\w.\-]+)+|(?:[\w.\-]+/)*[\w\-]+(?:\.[\w\-]+)*\.(?:%s))(?![\w])"
    % "|".join(sorted(set(DELIVERABLE_EXTS))), re.IGNORECASE)


def is_absolute_path(p: str) -> bool:
    p = p.strip().strip("`'\"")
    return p.startswith("/") or p.startswith("\\") or (len(p) >= 2 and p[1] == ":") or os.path.isabs(p)


def resolve_path(workdir: str, p: str) -> str:
    p = p.strip().strip("`'\"")
    if is_absolute_path(p):
        if p.startswith("/app/") and workdir not in ("/app", "/app/"):
            rel = p[5:]
            return (Path(workdir) / rel).as_posix()
        return Path(p).as_posix()
    return (Path(workdir) / p).as_posix()


def extract_candidate_paths(instruction: str, workdir: str = "/app") -> list[str]:
    """Absolute paths of files mentioned in the task text (inputs *and* outputs)."""
    seen: dict[str, None] = {}
    for m in _PATH_RE.finditer(instruction or ""):
        tok = m.group(1).rstrip(".,;:)")
        base = posixpath.basename(tok)
        if "." not in base:
            continue  # bare directories like /app/data
        if "." in base and base.rsplit(".", 1)[1].lower() not in DELIVERABLE_EXTS:
            continue
        if re.fullmatch(r"\d+\.\d+", base):
            continue
        seen[resolve_path(workdir, tok)] = None
    return list(seen)


AUDIT_SCRIPT = r'''
import ast, csv, json, math, os, re, sys, zipfile
args = json.load(open(sys.argv[1]))
BAD_TOKENS = {"nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"}
EMPTYISH = {"null", "none", ""}

def walk_bad(o, path="$", out=None, limit=10):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        out.append(path + "=" + repr(o))
    elif isinstance(o, dict):
        for k, v in o.items():
            walk_bad(v, path + "." + str(k), out, limit)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            walk_bad(v, path + "[%d]" % i, out, limit)
    return out

def _no_const(c):
    raise ValueError("non-finite constant %s" % c)

def check(path):
    r = {"path": path, "exists": False, "size": 0, "ok": False, "issues": [], "warnings": []}
    if not os.path.exists(path):
        r["issues"].append("missing file"); return r
    r["exists"] = True
    if os.path.isdir(path):
        n = sum(len(f) for _, _, f in os.walk(path))
        r["size"] = n; r["ok"] = n > 0
        if n == 0: r["issues"].append("empty directory")
        return r
    size = os.path.getsize(path); r["size"] = size
    if size == 0:
        r["issues"].append("zero-byte file"); return r
    ext = path.rsplit(".", 1)[-1].lower() if "." in os.path.basename(path) else ""
    iss, warn = r["issues"], r["warnings"]
    try:
        if ext == "json":
            txt = open(path, encoding="utf-8-sig").read()
            try:
                obj = json.loads(txt, parse_constant=_no_const)
            except ValueError as e:
                iss.append("JSON invalid/non-finite: %s" % e); obj = None
            if obj is not None:
                bad = walk_bad(obj)
                if bad: iss.append("non-finite numbers: " + ", ".join(bad))
                if obj in ({}, [], None): warn.append("JSON is empty")
        elif ext in ("yaml", "yml"):
            try:
                import yaml
                obj = yaml.safe_load(open(path, encoding="utf-8"))
                bad = walk_bad(obj)
                if bad: iss.append("non-finite numbers: " + ", ".join(bad))
                if obj is None: warn.append("YAML is empty")
            except ImportError:
                warn.append("pyyaml unavailable in container; YAML parse skipped")
            except Exception as e:
                iss.append("YAML invalid: %s" % str(e)[:200])
        elif ext in ("csv", "tsv"):
            delim = "\t" if ext == "tsv" else ","
            rows = list(csv.reader(open(path, encoding="utf-8-sig", newline=""), delimiter=delim))
            if len(rows) < 1: iss.append("no rows")
            else:
                w = len(rows[0]); badc = []; emp = 0
                if len(rows) < 2: warn.append("header only, no data rows")
                for ri, row in enumerate(rows):
                    if len(row) != w and any(c.strip() for c in row):
                        warn.append("row %d has %d cols (header %d)" % (ri + 1, len(row), w)); break
                for ri, row in enumerate(rows[1:], 2):
                    for ci, c in enumerate(row):
                        t = c.strip().lower()
                        if t in BAD_TOKENS and len(badc) < 8: badc.append("r%dc%d=%s" % (ri, ci + 1, c))
                        elif t in EMPTYISH: emp += 1
                if badc: iss.append("NaN/Inf cells: " + ", ".join(badc))
                if emp: warn.append("%d empty/null cells" % emp)
        elif ext in ("xlsx", "xlsm"):
            if not zipfile.is_zipfile(path): iss.append("not a valid xlsx zip")
            else:
                try:
                    import openpyxl
                    wbv = openpyxl.load_workbook(path, data_only=True)
                    wbf = openpyxl.load_workbook(path, data_only=False)
                    badc = []; nocache = 0; errs = []
                    for ws in wbv.worksheets:
                        wf = wbf[ws.title]
                        for row in ws.iter_rows():
                            for c in row:
                                v = c.value
                                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) and len(badc) < 8:
                                    badc.append("%s!%s=%r" % (ws.title, c.coordinate, v))
                                elif isinstance(v, str) and v.startswith("#") and v.rstrip("!?").upper() in ("#DIV/0", "#N/A", "#VALUE", "#REF", "#NAME", "#NUM", "#NULL") and len(errs) < 8:
                                    errs.append("%s!%s=%s" % (ws.title, c.coordinate, v))
                                elif v is None:
                                    f = wf[c.coordinate].value
                                    if isinstance(f, str) and f.startswith("="): nocache += 1
                    if badc: iss.append("NaN/Inf cells: " + ", ".join(badc))
                    if errs: iss.append("Excel error cells: " + ", ".join(errs))
                    if nocache: warn.append("%d formula cells have no cached value (graders reading values see None; write numbers or recalc with LibreOffice)" % nocache)
                    if not any(ws.max_row > 1 or ws.max_column > 1 or ws["A1"].value is not None for ws in wbv.worksheets):
                        iss.append("workbook is empty")
                except ImportError:
                    warn.append("openpyxl unavailable; only zip integrity checked")
                except Exception as e:
                    iss.append("XLSX unreadable: %s" % str(e)[:200])
        elif ext == "dxf":
            txt = open(path, encoding="utf-8", errors="replace").read()
            toks = [t.strip() for t in txt.splitlines()]
            if "SECTION" not in toks: iss.append("DXF has no SECTION")
            if not toks or toks[-1] != "EOF" and "EOF" not in toks[-4:]: iss.append("DXF missing EOF")
            try:
                import ezdxf
                ezdxf.readfile(path)
            except ImportError:
                pass
            except Exception as e:
                iss.append("ezdxf cannot read: %s" % str(e)[:200])
        elif ext == "ifc":
            txt = open(path, encoding="utf-8", errors="replace").read()
            if not txt.lstrip().startswith("ISO-10303-21"): iss.append("IFC missing ISO-10303-21 header")
            if "ENDSEC;" not in txt or "END-ISO-10303-21" not in txt: iss.append("IFC missing ENDSEC/END-ISO-10303-21")
            if "DATA;" not in txt: iss.append("IFC has no DATA section")
        elif ext in ("md", "markdown"):
            txt = open(path, encoding="utf-8", errors="replace").read()
            if not re.search(r"^#{1,6}\s+\S", txt, re.M): warn.append("markdown has no headings")
            if re.search(r"\b(TODO|TBD|FIXME|lorem ipsum)\b", txt, re.I): warn.append("placeholder text (TODO/TBD) present")
            if len(txt.strip()) < 40: warn.append("very short report")
        elif ext == "py":
            try: compile(open(path, encoding="utf-8").read(), path, "exec")
            except SyntaxError as e: iss.append("Python syntax error: %s" % e)
        elif ext == "pdf":
            if not open(path, "rb").read(5).startswith(b"%PDF"): iss.append("not a PDF")
        elif ext == "docx":
            if not zipfile.is_zipfile(path): iss.append("not a valid docx zip")
        elif ext == "txt":
            if not open(path, encoding="utf-8", errors="replace").read().strip(): iss.append("blank text file")
    except Exception as e:
        iss.append("audit error: %s" % str(e)[:200])
    r["ok"] = not iss
    return r

res = [check(p) for p in args["paths"]]
print("APEX_AUDIT_JSON:" + json.dumps({"ok": all(x["ok"] for x in res), "files": res}))
'''


CHECKPOINT_SCRIPT = r'''
import hashlib, json, os, shutil, sys

action = sys.argv[1]
ck_dir = sys.argv[2]

if action == "create":
    args = json.load(open(sys.argv[3], encoding="utf-8"))
    paths = args.get("paths", [])
    workdir = args.get("workdir", "")
    label = args.get("label", "G1")
    new_dir = ck_dir + ".new"
    files_dir = os.path.join(new_dir, "files")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir, ignore_errors=True)
    os.makedirs(files_dir, exist_ok=True)
    count = 0
    hashes = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            rel = os.path.relpath(p, workdir) if workdir and os.path.commonpath([p, workdir]) == workdir else os.path.basename(p)
        except Exception:
            rel = os.path.basename(p)
        dest = os.path.join(files_dir, rel)
        if os.path.isdir(p):
            shutil.copytree(p, dest, dirs_exist_ok=True)
            for root, _, fs in os.walk(p):
                count += len(fs)
        elif os.path.isfile(p):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(p, dest)
            count += 1
            try:
                with open(p, "rb") as f:
                    hashes[rel] = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                pass
    with open(os.path.join(new_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, "paths": paths, "workdir": workdir, "hashes": hashes}, f)
    prev_dir = ck_dir + ".prev"
    if os.path.exists(prev_dir):
        shutil.rmtree(prev_dir, ignore_errors=True)
    if os.path.exists(ck_dir):
        os.rename(ck_dir, prev_dir)
    os.rename(new_dir, ck_dir)
    print("APEX_CK_OK:%d" % count)

elif action == "restore":
    files_dir = os.path.join(ck_dir, "files")
    meta_path = os.path.join(ck_dir, "meta.json")
    if not os.path.isdir(files_dir):
        print("NO_CHECKPOINT")
        sys.exit(0)
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.isfile(meta_path) else {}
    workdir = meta.get("workdir", "")
    restored = 0
    for root, _, files in os.walk(files_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, files_dir)
            target = os.path.join(workdir, rel) if workdir else ("/" + rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(src, target)
            restored += 1
    print("APEX_CK_RESTORED:%d" % restored)
'''


class Auditor:
    def __init__(self, protocol: PLEProtocol):
        self.protocol = protocol
        self.last_audit: dict | None = None
        self.last_checkpoint: dict | None = None

    # ------------------------------------------------------------- audit
    async def audit_deliverables(self, workdir: str, expected_files: Iterable[str]) -> dict:
        paths = [resolve_path(workdir, p) for p in expected_files]
        result: dict | None = None
        try:
            await self.protocol.write_file(AUDIT_SCRIPT_PATH, AUDIT_SCRIPT)
            await self.protocol.write_file(AUDIT_ARGS_PATH, json.dumps({"paths": paths}))
            out = await self.protocol.exec_command(
                f"(python3 {AUDIT_SCRIPT_PATH} {AUDIT_ARGS_PATH} || python {AUDIT_SCRIPT_PATH} {AUDIT_ARGS_PATH})",
                timeout_sec=120)
            for line in out["stdout"].splitlines():
                if line.startswith("APEX_AUDIT_JSON:"):
                    result = json.loads(line[len("APEX_AUDIT_JSON:"):])
                    break
            if result is None:
                log.warning("audit script produced no result: %s", out["stderr"][:300])
        except Exception as e:
            log.warning("in-container audit failed: %r", e)
        if result is None:
            result = await self._shell_fallback(paths)
        result["summary"] = self.format_summary(result)
        self.last_audit = result
        return result

    async def _shell_fallback(self, paths: list[str]) -> dict:
        files = []
        for p in paths:
            q = shlex.quote(p)
            try:
                out = await self.protocol.exec_command(f"test -s {q} && echo OK || echo BAD", 30)
                ok = out["stdout"].strip().endswith("OK")
            except Exception:
                ok = False
            files.append({"path": p, "exists": ok, "ok": ok, "issues": [] if ok else ["missing or empty"],
                          "warnings": ["python unavailable: shallow check only"]})
        return {"ok": all(f["ok"] for f in files), "files": files}

    @staticmethod
    def format_summary(result: dict) -> str:
        lines = []
        for f in result.get("files", []):
            tag = "PASS" if f["ok"] else "FAIL"
            extra = "; ".join(f.get("issues", []) + [f"warn: {w}" for w in f.get("warnings", [])])
            lines.append(f"- [{tag}] {f['path']} ({f.get('size', 0)} bytes)" + (f" - {extra}" if extra else ""))
        head = "AUDIT PASSED" if result.get("ok") else "AUDIT FAILED"
        return head + "\n" + "\n".join(lines)

    # ------------------------------------------------------------- checkpoint
    async def create_checkpoint(self, workdir: str, deliverables: Iterable[str],
                                label: str = "G1") -> dict:
        """Snapshot deliverables to /tmp/.apex_ple_checkpoint (atomic swap, previous kept as .prev)."""
        paths = [resolve_path(workdir, p) for p in deliverables]
        ck = CHECKPOINT_DIR
        meta = {"label": label, "paths": paths, "workdir": workdir}
        # 1. Primary: portable Python script (works across Linux containers & Windows mock tests)
        try:
            await self.protocol.write_file(CHECKPOINT_SCRIPT_PATH, CHECKPOINT_SCRIPT)
            await self.protocol.write_file(CHECKPOINT_ARGS_PATH, json.dumps(meta))
            cmd = f"(python3 {CHECKPOINT_SCRIPT_PATH} create {ck} {CHECKPOINT_ARGS_PATH} || python {CHECKPOINT_SCRIPT_PATH} create {ck} {CHECKPOINT_ARGS_PATH})"
            out = await self.protocol.exec_command(cmd, timeout_sec=120)
            for line in out["stdout"].splitlines():
                if line.startswith("APEX_CK_OK:"):
                    n_files = int(line[len("APEX_CK_OK:"):])
                    self.last_checkpoint = {
                        "ok": n_files > 0, "files": n_files, "paths": paths,
                        "label": label, "stderr": out["stderr"][:300]
                    }
                    return self.last_checkpoint
        except Exception as e:
            log.warning("python checkpoint create failed: %r; attempting shell fallback", e)

        # 2. Fallback: POSIX shell commands
        lines = [f"rm -rf {ck}.new && mkdir -p {ck}.new/files"]
        for p in paths:
            q = shlex.quote(p)
            dest = shlex.quote(f"{ck}.new/files{p}")
            ddir = shlex.quote(f"{ck}.new/files{posixpath.dirname(p)}")
            lines.append(
                f"if [ -d {q} ]; then mkdir -p {dest} && cp -a {q}/. {dest}/;"
                f" elif [ -f {q} ]; then mkdir -p {ddir} && cp -p {q} {dest}; fi")
        lines.append(f"(cd {ck}.new/files && find . -type f -exec sha256sum {{}} + 2>/dev/null) > {ck}.new/sha256.txt || true")
        lines.append(f"echo {shlex.quote(json.dumps(meta))} > {ck}.new/meta.json")
        lines.append(f"rm -rf {ck}.prev; [ -d {ck} ] && mv {ck} {ck}.prev; mv {ck}.new {ck}")
        lines.append(f"find {ck}/files -type f | wc -l")
        out = await self.protocol.exec_command("\n".join(lines), timeout_sec=120)
        ok = out["exit_code"] == 0
        try:
            n_files = int(out["stdout"].strip().splitlines()[-1])
        except Exception:
            n_files = -1
        self.last_checkpoint = {"ok": ok and n_files != 0, "files": n_files, "paths": paths,
                                "label": label, "stderr": out["stderr"][:300]}
        return self.last_checkpoint

    async def restore_checkpoint(self) -> dict:
        """Restore the last certified checkpoint over the working files."""
        ck = CHECKPOINT_DIR
        # 1. Primary: portable Python script
        try:
            await self.protocol.write_file(CHECKPOINT_SCRIPT_PATH, CHECKPOINT_SCRIPT)
            cmd = f"(python3 {CHECKPOINT_SCRIPT_PATH} restore {ck} || python {CHECKPOINT_SCRIPT_PATH} restore {ck})"
            out = await self.protocol.exec_command(cmd, timeout_sec=120)
            if "APEX_CK_RESTORED" in out["stdout"]:
                return {"restored": True, "stderr": out["stderr"][:300]}
            if "NO_CHECKPOINT" in out["stdout"]:
                return {"restored": False, "stderr": "No checkpoint directory"}
        except Exception as e:
            log.warning("python checkpoint restore failed: %r; attempting shell fallback", e)

        # 2. Fallback: POSIX shell commands
        out = await self.protocol.exec_command(
            f"if [ -d {ck}/files ]; then cp -a {ck}/files/. / && echo RESTORED; else echo NO_CHECKPOINT; fi",
            timeout_sec=120)
        return {"restored": "RESTORED" in out["stdout"], "stderr": out["stderr"][:300]}

    async def checkpoint_if_valid(self, workdir: str, deliverables: list[str]) -> dict:
        """Audit first; checkpoint only if the audit passes (so the checkpoint is always G=1-certified)."""
        audit = await self.audit_deliverables(workdir, deliverables)
        if not audit["ok"]:
            return {"checkpointed": False, "audit": audit}
        ck = await self.create_checkpoint(workdir, deliverables)
        return {"checkpointed": ck["ok"], "audit": audit, "checkpoint": ck}
