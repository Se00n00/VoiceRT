"""Operator console backend for /talk and /chat.

- ``GET /console/snapshot`` — recent events, active calls, uptime.
- ``WS /console/stream`` — history replay, then live tail.
- Terminal UI: ``PYTHONPATH=. .venv/bin/python -m src.server.console_cli``.
"""

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import state
from ..state import console_history, console_subscribe, console_unsubscribe

router = APIRouter()


@router.get("/console/snapshot")
async def console_snapshot():
    calls = []
    try:
        from .talk import _active_call

        call = _active_call.get("call")
        if call is not None and not getattr(call, "done", True):
            calls.append({"sid": getattr(call, "sid", ""), "state": getattr(call, "state", "")})
    except Exception:
        pass
    return {
        "uptime_s": round(time.time() - state._t_boot, 1),
        "turns_done": state._turns["done"],
        "active_calls": calls,
        "events": console_history(),
    }


@router.websocket("/console/stream")
async def console_stream(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()
    console_subscribe(queue)
    try:
        for ev in console_history():
            try:
                await ws.send_json(ev)
            except Exception:
                return
        while True:
            try:
                ev = await queue.get()
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                continue
            try:
                await ws.send_json(ev)
            except Exception:
                break
    finally:
        console_unsubscribe(queue)
        try:
            await ws.close()
        except Exception:
            pass
