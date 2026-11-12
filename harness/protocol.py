"""PLE Protocol v1 wire layer (async WebSocket).

The harness is the WebSocket *server*; the PLE evaluator connects, sends one
``task`` frame, then answers every action frame we emit (``exec``, ``read_file``,
``write_file``, ``http``) with a result frame carrying the same ``id``. We finish
the session by sending a single ``done`` frame.

Result frames are matched purely on ``id`` so the exact result ``type`` string
(``exec_result``, ``result`` ...) used by the evaluator does not matter.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("apex.protocol")

DEFAULT_LIMITS = {
    "agent_timeout_sec": 21600,
    "exec_timeout_sec": 600,
    "max_output_bytes": 100000,
    "max_write_bytes": 10000000,
}


class ProtocolError(RuntimeError):
    """Raised on a wire-level failure (connection closed, bad frame, ...)."""


class ConnectionLost(ProtocolError):
    pass


@dataclass
class Limits:
    agent_timeout_sec: int = DEFAULT_LIMITS["agent_timeout_sec"]
    exec_timeout_sec: int = DEFAULT_LIMITS["exec_timeout_sec"]
    max_output_bytes: int = DEFAULT_LIMITS["max_output_bytes"]
    max_write_bytes: int = DEFAULT_LIMITS["max_write_bytes"]

    @classmethod
    def from_dict(cls, d: dict | None) -> "Limits":
        d = d or {}
        out = cls()
        for k in DEFAULT_LIMITS:
            try:
                v = d.get(k)
                if v is not None and float(v) > 0:
                    setattr(out, k, int(float(v)))
            except (TypeError, ValueError):
                pass
        return out


@dataclass
class MCPServerSpec:
    name: str
    url: str
    transport: str = "streamable-http"


@dataclass
class TaskSpec:
    protocol: int = 1
    session: str = ""
    instruction: str = ""
    workdir: str = "/app"
    limits: Limits = field(default_factory=Limits)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_frame(cls, msg: dict) -> "TaskSpec":
        servers = []
        for i, s in enumerate(msg.get("mcp_servers") or []):
            if isinstance(s, str):
                s = {"name": f"server{i}", "url": s}
            if not isinstance(s, dict) or not s.get("url"):
                continue
            servers.append(MCPServerSpec(
                name=str(s.get("name") or f"server{i}"),
                url=str(s["url"]),
                transport=str(s.get("transport") or "streamable-http"),
            ))
        return cls(
            protocol=int(msg.get("protocol") or 1),
            session=str(msg.get("session") or ""),
            instruction=str(msg.get("instruction") or ""),
            workdir=str(msg.get("workdir") or "/app"),
            limits=Limits.from_dict(msg.get("limits")),
            mcp_servers=servers,
            raw=msg,
        )


def truncate_head_tail(data: bytes | str, max_bytes: int) -> tuple[str, bool]:
    """Keep head+tail of ``data`` within ~``max_bytes``; returns (text, truncated)."""
    raw = data.encode("utf-8", "replace") if isinstance(data, str) else data
    if max_bytes <= 0 or len(raw) <= max_bytes:
        return raw.decode("utf-8", "replace"), False
    marker_room = 120
    keep = max(max_bytes - marker_room, 64)
    head_n = int(keep * 0.6)
    tail_n = keep - head_n
    omitted = len(raw) - head_n - tail_n
    head = raw[:head_n].decode("utf-8", "ignore")
    tail = raw[-tail_n:].decode("utf-8", "ignore")
    return f"{head}\n...[{omitted} bytes truncated]...\n{tail}", True


def _b64d(s: str | None) -> bytes:
    if not s:
        return b""
    s = s + "=" * (-len(s) % 4)
    return base64.b64decode(s)


class PLEProtocol:
    """Async request/response multiplexer over one evaluator WebSocket."""

    def __init__(self, ws: Any, *, start_time: float | None = None):
        self.ws = ws
        self.task: TaskSpec | None = None
        self.start_time = start_time if start_time is not None else time.monotonic()
        self._pending: dict[str, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._closed = False
        self.done_sent = False
        self.cancel_requested = False
        self._send_lock = asyncio.Lock()
        self.stats = {"exec": 0, "read_file": 0, "write_file": 0, "http": 0}

    # ------------------------------------------------------------------ setup
    async def receive_task(self, timeout: float | None = 120.0) -> TaskSpec:
        """Wait for the inbound ``task`` frame, then start the result reader."""
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout)
            msg = self._decode(raw)
            if msg is None:
                continue
            if msg.get("type") == "task":
                break
            log.warning("ignoring pre-task frame type=%r", msg.get("type"))
        self.task = TaskSpec.from_frame(msg)
        self.start_time = time.monotonic()
        self._reader = asyncio.ensure_future(self._read_loop())
        log.info("task received session=%s workdir=%s limits=%s mcp=%s",
                 self.task.session, self.task.workdir, self.task.limits,
                 [s.name for s in self.task.mcp_servers])
        return self.task

    @property
    def limits(self) -> Limits:
        return self.task.limits if self.task else Limits()

    @staticmethod
    def _decode(raw: Any) -> dict | None:
        if isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw).decode("utf-8", "replace")
        try:
            msg = json.loads(raw)
        except Exception:
            log.warning("non-JSON frame dropped: %r", str(raw)[:200])
            return None
        return msg if isinstance(msg, dict) else None

    async def _read_loop(self) -> None:
        try:
            async for raw in self.ws:
                msg = self._decode(raw)
                if msg is None:
                    continue
                mtype = msg.get("type")
                if mtype in ("cancel", "abort", "stop"):
                    self.cancel_requested = True
                    log.warning("evaluator requested %s", mtype)
                    continue
                rid = msg.get("id")
                fut = self._pending.pop(str(rid), None) if rid is not None else None
                if fut is not None and not fut.done():
                    fut.set_result(msg)
                else:
                    log.debug("unmatched frame type=%r id=%r", mtype, rid)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # connection dropped
            log.warning("reader terminated: %r", e)
        finally:
            self._closed = True
            err = ConnectionLost("websocket closed")
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()

    async def close(self) -> None:
        if self._reader and not self._reader.done():
            self._reader.cancel()
            try:
                await self._reader
            except BaseException:
                pass

    # ------------------------------------------------------------------ budget
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.agent_timeout_sec - self.elapsed_seconds())

    def is_near_timeout(self, ratio: float = 0.88) -> bool:
        return self.elapsed_seconds() >= ratio * self.limits.agent_timeout_sec

    # ------------------------------------------------------------------ core RPC
    async def _send(self, frame: dict) -> None:
        if self._closed:
            raise ConnectionLost("websocket closed")
        async with self._send_lock:
            try:
                await self.ws.send(json.dumps(frame))
            except Exception as e:
                self._closed = True
                raise ConnectionLost(f"send failed: {e!r}") from e

    async def request(self, frame: dict, timeout: float) -> dict:
        rid = uuid.uuid4().hex[:16]
        frame = {**frame, "id": rid}
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._send(frame)
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise ProtocolError(f"{frame.get('type')} request timed out after {timeout}s")
        finally:
            self._pending.pop(rid, None)

    # ------------------------------------------------------------------ actions
    async def exec_command(self, command: str, timeout_sec: int = 120) -> dict:
        """Run a shell command in the task container.

        Returns ``{stdout, stderr, exit_code, timed_out, truncated, error}``.
        stdout/stderr are head+tail truncated to ``max_output_bytes`` combined.
        """
        self.stats["exec"] += 1
        timeout_sec = int(max(1, min(timeout_sec, self.limits.exec_timeout_sec)))
        msg = await self.request(
            {"type": "exec", "command": command, "timeout_sec": timeout_sec},
            timeout=timeout_sec + 60,
        )
        res = msg.get("result") if isinstance(msg.get("result"), dict) else msg
        stdout = res.get("stdout", "")
        stderr = res.get("stderr", "")
        if isinstance(stdout, (bytes, bytearray)):
            stdout = bytes(stdout).decode("utf-8", "replace")
        if isinstance(stderr, (bytes, bytearray)):
            stderr = bytes(stderr).decode("utf-8", "replace")
        if res.get("stdout_b64"):
            stdout = _b64d(res["stdout_b64"]).decode("utf-8", "replace")
        if res.get("stderr_b64"):
            stderr = _b64d(res["stderr_b64"]).decode("utf-8", "replace")
        code = res.get("exit_code", res.get("returncode", res.get("return_code")))
        budget = self.limits.max_output_bytes
        stdout_t, t1 = truncate_head_tail(stdout or "", int(budget * 0.7))
        stderr_t, t2 = truncate_head_tail(stderr or "", int(budget * 0.3))
        err = res.get("error") or msg.get("error")
        timed_out = bool(res.get("timed_out") or res.get("timeout"))
        if code is None:
            code = -1 if (err or timed_out) else 0
        return {
            "stdout": stdout_t, "stderr": stderr_t, "exit_code": code,
            "timed_out": timed_out, "truncated": t1 or t2, "error": err,
        }

    async def read_file(self, path: str) -> bytes:
        self.stats["read_file"] += 1
        msg = await self.request({"type": "read_file", "path": path},
                                 timeout=min(self.limits.exec_timeout_sec, 300) + 30)
        res = msg.get("result") if isinstance(msg.get("result"), dict) else msg
        if res.get("error"):
            raise FileNotFoundError(f"{path}: {res['error']}")
        if res.get("content_b64") is not None:
            return _b64d(res["content_b64"])
        content = res.get("content")
        if content is None:
            raise FileNotFoundError(f"{path}: empty read_file response")
        if isinstance(content, str):
            return content.encode("utf-8")
        if isinstance(content, list):
            return bytes(content)
        return bytes(content)

    async def write_file(self, path: str, content: bytes | str, append: bool = False) -> dict:
        """Write a file. Payloads above ``max_write_bytes`` are split into appends."""
        self.stats["write_file"] += 1
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        limit = max(1024, self.limits.max_write_bytes)
        chunks = [data[i:i + limit] for i in range(0, len(data), limit)] or [b""]
        res: dict = {}
        for i, chunk in enumerate(chunks):
            msg = await self.request(
                {"type": "write_file", "path": path,
                 "content_b64": base64.b64encode(chunk).decode("ascii"),
                 "append": bool(append or i > 0)},
                timeout=120)
            res = msg.get("result") if isinstance(msg.get("result"), dict) else msg
            if res.get("error"):
                return {"ok": False, "error": res["error"], "path": path}
        return {"ok": True, "path": path, "bytes": len(data)}

    async def http_request(self, method: str, url: str,
                           headers: dict | None = None,
                           body: bytes | str | None = None,
                           timeout: float = 120.0) -> dict:
        """Proxy an HTTP request inside the task container.

        Returns ``{status, headers, body(bytes), text, error}``.
        """
        self.stats["http"] += 1
        frame: dict[str, Any] = {"type": "http", "method": method.upper(),
                                 "url": url, "headers": headers or {}}
        if body is not None:
            raw = body.encode("utf-8") if isinstance(body, str) else bytes(body)
            frame["body_b64"] = base64.b64encode(raw).decode("ascii")
        else:
            frame["body_b64"] = ""
        frame["timeout_sec"] = int(timeout)
        msg = await self.request(frame, timeout=timeout + 30)
        res = msg.get("result") if isinstance(msg.get("result"), dict) else msg
        if res.get("body_b64") is not None:
            raw_body = _b64d(res["body_b64"])
        else:
            b = res.get("body", res.get("text", ""))
            raw_body = b.encode("utf-8") if isinstance(b, str) else bytes(b or b"")
        hdrs = res.get("headers") or {}
        status = res.get("status", res.get("status_code"))
        return {
            "status": int(status) if status is not None else 0,
            "headers": {str(k): v for k, v in hdrs.items()} if isinstance(hdrs, dict) else {},
            "body": raw_body,
            "text": raw_body.decode("utf-8", "replace"),
            "error": res.get("error"),
        }

    async def send_done(self, final_message: str) -> None:
        """Send the single terminal ``done`` frame (idempotent)."""
        if self.done_sent:
            return
        self.done_sent = True
        try:
            await self._send({"type": "done", "final_message": final_message})
            log.info("done frame sent (%d chars)", len(final_message))
        except ConnectionLost:
            log.error("could not deliver done frame: connection lost")
