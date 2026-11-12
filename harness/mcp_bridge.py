"""MCP Streamable-HTTP (JSON-RPC 2.0) client tunnelled through ``protocol.http_request``."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .protocol import MCPServerSpec, PLEProtocol

log = logging.getLogger("apex.mcp")

MCP_PROTOCOL_VERSION = "2025-03-26"
_NAME_BAD = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize(s: str) -> str:
    return _NAME_BAD.sub("_", s)


def parse_rpc_body(text: str, want_id: Any = None) -> dict | None:
    """Parse a JSON or SSE (``data:`` lines) response body into a JSON-RPC object."""
    text = (text or "").strip()
    if not text:
        return None
    candidates: list[dict] = []
    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)
            candidates = obj if isinstance(obj, list) else [obj]
        except Exception:
            pass
    if not candidates:
        data_lines: list[str] = []
        events: list[str] = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif not line.strip() and data_lines:
                events.append("\n".join(data_lines))
                data_lines = []
        if data_lines:
            events.append("\n".join(data_lines))
        for ev in events:
            try:
                o = json.loads(ev)
                candidates.extend(o if isinstance(o, list) else [o])
            except Exception:
                continue
    for c in candidates:
        if isinstance(c, dict) and (want_id is None or c.get("id") == want_id):
            return c
    return candidates[0] if candidates and isinstance(candidates[0], dict) else None


@dataclass
class MCPTool:
    server: str
    name: str
    description: str
    input_schema: dict
    fn_name: str = ""


@dataclass
class _ServerState:
    spec: MCPServerSpec
    session_id: str | None = None
    initialized: bool = False
    next_id: int = 1
    tools: list[MCPTool] = field(default_factory=list)
    error: str | None = None


class MCPBridge:
    def __init__(self, protocol: PLEProtocol, servers: list[MCPServerSpec] | None = None):
        self.protocol = protocol
        specs = servers if servers is not None else (protocol.task.mcp_servers if protocol.task else [])
        self.servers: dict[str, _ServerState] = {s.name: _ServerState(s) for s in specs}
        self._fn_map: dict[str, MCPTool] = {}

    # ------------------------------------------------------------- JSON-RPC
    async def _post(self, st: _ServerState, payload: dict, expect_response: bool = True) -> dict | None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        if st.session_id:
            headers["Mcp-Session-Id"] = st.session_id
        resp = await self.protocol.http_request("POST", st.spec.url, headers, json.dumps(payload))
        for k, v in resp["headers"].items():
            if k.lower() == "mcp-session-id" and v:
                st.session_id = v if isinstance(v, str) else str(v[0])
        if resp.get("error"):
            raise RuntimeError(f"http proxy error: {resp['error']}")
        if not expect_response:
            return None
        if resp["status"] >= 400 or resp["status"] == 0:
            raise RuntimeError(f"MCP HTTP {resp['status']}: {resp['text'][:300]}")
        obj = parse_rpc_body(resp["text"], payload.get("id"))
        if obj is None:
            raise RuntimeError(f"unparseable MCP response: {resp['text'][:300]!r}")
        if "error" in obj and obj["error"]:
            e = obj["error"]
            raise RuntimeError(f"MCP error {e.get('code')}: {e.get('message')}" if isinstance(e, dict) else str(e))
        return obj

    async def _rpc(self, st: _ServerState, method: str, params: dict | None = None) -> Any:
        rid = st.next_id
        st.next_id += 1
        payload: dict = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            payload["params"] = params
        obj = await self._post(st, payload)
        return (obj or {}).get("result")

    async def _initialize(self, st: _ServerState) -> None:
        await self._rpc(st, "initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "apex-ple-harness", "version": "1.0.0"},
        })
        try:
            await self._post(st, {"jsonrpc": "2.0", "method": "notifications/initialized"},
                             expect_response=False)
        except Exception as e:  # notification failures are non-fatal
            log.debug("initialized notification failed: %r", e)
        st.initialized = True

    # ------------------------------------------------------------- discovery
    async def discover_tools(self) -> list[MCPTool]:
        """``initialize`` + ``tools/list`` on every declared server."""
        self._fn_map.clear()
        all_tools: list[MCPTool] = []
        for st in self.servers.values():
            st.tools, st.error = [], None
            try:
                await self._initialize(st)
                cursor = None
                while True:
                    res = await self._rpc(st, "tools/list", {"cursor": cursor} if cursor else {}) or {}
                    for t in res.get("tools", []):
                        tool = MCPTool(st.spec.name, t.get("name", ""), t.get("description", "") or "",
                                       t.get("inputSchema") or {"type": "object", "properties": {}})
                        st.tools.append(tool)
                    cursor = res.get("nextCursor")
                    if not cursor:
                        break
            except Exception as e:
                st.error = repr(e)
                log.warning("MCP discovery failed for %s: %s", st.spec.name, e)
            for tool in st.tools:
                fn = f"mcp__{_sanitize(tool.server)}__{_sanitize(tool.name)}"
                if len(fn) > 64:
                    fn = fn[:64]
                base, n = fn, 2
                while fn in self._fn_map:
                    fn = f"{base[:60]}_{n}"
                    n += 1
                tool.fn_name = fn
                self._fn_map[fn] = tool
                all_tools.append(tool)
        return all_tools

    def openai_tools(self) -> list[dict]:
        out = []
        for fn, t in self._fn_map.items():
            schema = t.input_schema if isinstance(t.input_schema, dict) else {}
            schema = {"type": "object", "properties": {}, **schema}
            out.append({"type": "function", "function": {
                "name": fn,
                "description": f"[MCP server '{t.server}'] {t.description}"[:1024],
                "parameters": schema,
            }})
        return out

    def summary(self) -> dict:
        return {name: {"url": st.spec.url, "tools": [t.name for t in st.tools], "error": st.error}
                for name, st in self.servers.items()}

    def owns(self, fn_name: str) -> bool:
        return fn_name in self._fn_map

    # ------------------------------------------------------------- invocation
    async def call_tool(self, server_name: str, tool_name: str, arguments: dict | None = None) -> str:
        st = self.servers.get(server_name)
        if st is None:
            return f"ERROR: unknown MCP server {server_name!r}"
        try:
            if not st.initialized:
                await self._initialize(st)
            try:
                res = await self._rpc(st, "tools/call", {"name": tool_name, "arguments": arguments or {}})
            except RuntimeError as e:
                if "404" in str(e) or "session" in str(e).lower():  # stale session -> re-init once
                    st.session_id, st.initialized = None, False
                    await self._initialize(st)
                    res = await self._rpc(st, "tools/call", {"name": tool_name, "arguments": arguments or {}})
                else:
                    raise
        except Exception as e:
            return f"ERROR calling {server_name}/{tool_name}: {e}"
        return self._render_result(res or {})

    async def call_function(self, fn_name: str, arguments: dict | None) -> str:
        t = self._fn_map.get(fn_name)
        if t is None:
            return f"ERROR: unknown MCP tool {fn_name!r}"
        return await self.call_tool(t.server, t.name, arguments)

    @staticmethod
    def _render_result(res: dict) -> str:
        parts: list[str] = []
        for c in res.get("content", []) or []:
            if not isinstance(c, dict):
                parts.append(str(c))
            elif c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "resource":
                r = c.get("resource", {})
                parts.append(r.get("text") or f"[resource {r.get('uri')}]")
            else:
                parts.append(f"[{c.get('type', 'content')} omitted]")
        if not parts and res.get("structuredContent") is not None:
            parts.append(json.dumps(res["structuredContent"]))
        text = "\n".join(parts) if parts else json.dumps(res)
        return ("ERROR (tool reported isError): " + text) if res.get("isError") else text
