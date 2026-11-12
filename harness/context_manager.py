"""Token-aware conversation manager for multi-hour runs.

Layout of the request sent to the model::

    [system prompt]
    [PINNED  user: task instruction]
    [PINNED  user: deliverable manifest + latest G=1 checkpoint status + phase]
    [DIGEST  user: deterministic summary of compacted middle turns]
    [recent verbatim turns ...]

The middle of the transcript is compacted when the estimate exceeds the
threshold; assistant tool_calls and their ``tool`` replies are always kept or
dropped together so the API never sees an orphaned tool message.
"""
from __future__ import annotations

import json
from typing import Any

CHARS_PER_TOKEN = 3.3  # conservative for code / numeric-heavy text


def estimate_tokens(obj: Any) -> int:
    if obj is None:
        return 0
    if isinstance(obj, str):
        return int(len(obj) / CHARS_PER_TOKEN) + 1
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return int(len(s) / CHARS_PER_TOKEN) + 4


def _clip(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    h = int(n * 0.6)
    return s[:h] + f" ...[{len(s) - n} chars elided]... " + s[-(n - h):]


class ContextManager:
    def __init__(self, system_prompt: str, instruction: str, *,
                 max_context_tokens: int = 180_000, compact_ratio: float = 0.70,
                 keep_recent: int = 14, max_tool_chars: int = 24_000,
                 max_digest_chars: int = 24_000):
        self.system_prompt = system_prompt
        self.instruction = instruction
        self.max_context_tokens = max_context_tokens
        self.compact_ratio = compact_ratio
        self.keep_recent = keep_recent
        self.max_tool_chars = max_tool_chars
        self.max_digest_chars = max_digest_chars
        self.manifest = ""
        self.checkpoint_status = "No G=1 checkpoint yet."
        self.phase = "Phase 1: reconnaissance"
        self.digest_lines: list[str] = []
        self.history: list[dict] = []
        self.compactions = 0

    # ---------------------------------------------------------------- pinned
    def set_manifest(self, text: str) -> None:
        self.manifest = text

    def set_checkpoint_status(self, text: str) -> None:
        self.checkpoint_status = text

    def set_phase(self, text: str) -> None:
        self.phase = text

    def _pinned_state(self) -> dict:
        return {"role": "user", "content": (
            "[HARNESS STATE - pinned]\n"
            f"Current phase: {self.phase}\n"
            f"Deliverable manifest:\n{self.manifest or '(not yet determined)'}\n"
            f"Latest G=1 checkpoint status: {self.checkpoint_status}")}

    # ---------------------------------------------------------------- history
    def add(self, message: dict) -> None:
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            message = {**message, "content": _clip(message["content"], self.max_tool_chars)}
        self.history.append(message)

    def _fixed_messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.instruction},
                self._pinned_state()]
        if self.digest_lines:
            msgs.append({"role": "user", "content":
                         "[DIGEST of earlier work - compacted, details are on disk in the container]\n"
                         + "\n".join(self.digest_lines)})
        return msgs

    def estimate(self) -> int:
        return estimate_tokens(self._fixed_messages()) + estimate_tokens(self.history)

    def build(self) -> list[dict]:
        if self.estimate() > self.max_context_tokens * self.compact_ratio:
            self.compact()
        return self._fixed_messages() + list(self.history)

    # ---------------------------------------------------------------- compaction
    def _safe_cut(self, desired: int) -> int:
        """Largest cut <= desired where history[cut] is not a ``tool`` message."""
        cut = max(0, min(desired, len(self.history)))
        while 0 < cut < len(self.history) and self.history[cut].get("role") == "tool":
            cut -= 1
        return cut

    def _digest_of(self, m: dict) -> str | None:
        role = m.get("role")
        if role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                bits = []
                for c in calls:
                    f = c.get("function", {})
                    bits.append(f"{f.get('name')}({_clip(str(f.get('arguments', '')), 220)})")
                return "A> " + "; ".join(bits)
            return "A> " + _clip(m.get("content") or "", 300)
        if role == "tool":
            return "   <- " + _clip(str(m.get("content", "")).replace("\n", " "), 240)
        if role == "user":
            return "U> " + _clip(str(m.get("content", "")).replace("\n", " "), 300)
        return None

    def compact(self, aggressive: bool = False) -> bool:
        keep = max(4, self.keep_recent // (3 if aggressive else 1))
        cut = self._safe_cut(len(self.history) - keep)
        if cut <= 0 and not aggressive:
            return self._shrink_recent()
        for m in self.history[:cut]:
            d = self._digest_of(m)
            if d:
                self.digest_lines.append(d)
        self.history = self.history[cut:]
        # bound the digest, dropping oldest lines first
        total = sum(len(x) + 1 for x in self.digest_lines)
        limit = self.max_digest_chars // (2 if aggressive else 1)
        dropped = 0
        while total > limit and len(self.digest_lines) > 1:
            total -= len(self.digest_lines.pop(0)) + 1
            dropped += 1
        if dropped:
            self.digest_lines.insert(0, f"[... {dropped} oldest digest lines dropped ...]")
        self.compactions += 1
        if self.estimate() > self.max_context_tokens * self.compact_ratio:
            self._shrink_recent(aggressive)
        return True

    def _shrink_recent(self, aggressive: bool = False) -> bool:
        """Last resort: clip oversized verbatim messages inside the recent window."""
        limit = 2_000 if aggressive else 6_000
        changed = False
        for i, m in enumerate(self.history):
            c = m.get("content")
            if isinstance(c, str) and len(c) > limit:
                self.history[i] = {**m, "content": _clip(c, limit)}
                changed = True
        return changed

    def handle_context_error(self) -> bool:
        """Call after a provider 400 context-length error; returns True if state shrank."""
        before = self.estimate()
        self.max_context_tokens = int(self.max_context_tokens * 0.75)
        self.compact(aggressive=True)
        self._shrink_recent(True)
        return self.estimate() < before
