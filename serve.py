#!/usr/bin/env python3
"""Apex-PLE production entry point.

Listens on 0.0.0.0:$PORT (default 8080) for the PLE evaluator's WebSocket, runs
``ApexAgentLoop`` and *guarantees* one terminal ``done`` frame: on crash, hard
watchdog expiry, or cancellation the best G=1 checkpoint is restored first.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import websockets  # noqa: E402

from harness.agent_loop import ApexAgentLoop  # noqa: E402
from harness.protocol import PLEProtocol  # noqa: E402

log = logging.getLogger("apex.serve")

HARD_WATCHDOG_RATIO = 0.94  # agent loop self-limits at ~0.88; this is the last line of defence


async def handle_session(ws) -> None:
    proto = PLEProtocol(ws)
    loop_obj: ApexAgentLoop | None = None
    try:
        task = await proto.receive_task()
    except Exception as e:
        log.error("no valid task frame: %r", e)
        await proto.close()
        return
    loop_obj = ApexAgentLoop(proto, task)
    hard_limit = max(30.0, task.limits.agent_timeout_sec * HARD_WATCHDOG_RATIO)
    reason = None
    try:
        await asyncio.wait_for(loop_obj.run(), timeout=hard_limit)
    except asyncio.TimeoutError:
        reason = "hard watchdog timeout"
    except asyncio.CancelledError:
        reason = "cancelled"
        raise
    except Exception as e:
        log.exception("session crashed")
        reason = f"unexpected error {e!r}"
    finally:
        if not proto.done_sent:
            await _emergency_done(proto, loop_obj, reason or "terminated")
        await proto.close()


async def _emergency_done(proto: PLEProtocol, loop_obj: ApexAgentLoop, reason: str) -> None:
    """Restore the best checkpoint if needed, then emit ``done`` no matter what."""
    log.error("emergency finalisation: %s", reason)
    msg = ""
    try:
        async def _fin() -> str:
            # shield from the (already-cancelled) run: generous but bounded
            return await loop_obj.phase5_final()
        msg = await asyncio.wait_for(_fin(), timeout=90)
    except BaseException as e:  # includes timeouts/cancellation
        log.error("emergency phase5 failed: %r", e)
    if not msg:
        msg = (f"# Apex-PLE report\n\nSession ended early ({reason}). The best verified G=1 checkpoint "
               f"(if any) was kept; deliverables are in {proto.task.workdir if proto.task else '/app'}.")
    elif reason:
        msg += f"\n\n_Session note: ended via {reason}; last verified checkpoint was restored if the audit failed._"
    await proto.send_done(msg)


async def main() -> None:
    logging.basicConfig(level=getattr(logging, os.environ.get("LOG_LEVEL", "info").upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    try:  # websockets >= 13 has the new asyncio server; fall back to legacy API
        from websockets.asyncio.server import serve  # type: ignore
    except ImportError:  # pragma: no cover
        serve = websockets.serve  # type: ignore
    stop = asyncio.Event()
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
    except (NotImplementedError, RuntimeError):  # Windows
        pass
    async with serve(handle_session, host, port, max_size=None, ping_interval=20, ping_timeout=300):
        log.info("apex-ple-harness listening on ws://%s:%d (offline=%s)", host, port,
                 os.environ.get("APEX_OFFLINE_SOLVER", "0"))
        await stop.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
