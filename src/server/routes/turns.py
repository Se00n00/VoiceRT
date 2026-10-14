"""Agent turn sockets: gated terminal turns and autonomous turns."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..state import _maybe_title, _seen_sids, _wav_b64, get_agent, note_turn_end, note_turn_start

router = APIRouter()


def _event_json(event) -> dict:
    """Bridge event -> WS JSON, extracting wav bytes to b64.

    ..
    """
    d = dict(event.data or {})
    wav = d.pop("wav", None)
    if wav is not None:
        d["wav_b64"] = _wav_b64(wav)
    return {"event": event.kind, **d}


async def _send_cancelled(ws: WebSocket) -> None:
    """Best-effort cancelled notice. Never raises.

    ..
    """
    try:
        await ws.send_json({"event": "cancelled"})
    except Exception:
        pass


@router.websocket("/term")
async def term(ws: WebSocket) -> None:
    """Gated terminal turns with confirm pump and cancel.

    ..
    """
    await ws.accept()
    inbox: asyncio.Queue = asyncio.Queue()
    pending = {"event": None, "ok": False}
    turn_task = {"task": None}
    closed = {"done": False}

    def _cancel_turn() -> bool:
        """Cancel the running turn, releasing the confirm gate.

        ..
        """
        task = turn_task["task"]
        if task is None:
            return False
        pending["ok"] = False
        ev = pending["event"]
        if ev is not None:
            try:
                ev.set()
            except Exception:
                pass
        try:
            task.cancel()
        except Exception:
            pass
        return True

    async def confirm_fn(action) -> bool:
        pending["event"] = asyncio.Event()
        pending["ok"] = False
        try:
            await ws.send_json({"event": "confirm",
                                "action": action.as_dict()
                                if hasattr(action, "as_dict") else dict(action)})
        except Exception:
            pending["event"] = None
            return False
        try:
            await asyncio.wait_for(pending["event"].wait(), timeout=120)
        except asyncio.TimeoutError:
            pass
        finally:
            pending["event"] = None
        return bool(pending["ok"])

    harness = get_agent()

    async def run_one(text, sid, cwd) -> None:
        """Run one gated turn, streaming term events to the socket.

        ..
        """
        t0 = note_turn_start()
        was_cancelled = False
        try:
            async for event in harness.run_text(text, session_id=sid, cwd=cwd,
                                                confirm_fn=confirm_fn):
                if event.node != "term":
                    continue
                try:
                    await ws.send_json(_event_json(event))
                except Exception:
                    break
        except asyncio.CancelledError:
            was_cancelled = True
            await _send_cancelled(ws)
        except Exception as exc:
            try:
                await ws.send_json({"event": "error", "message": str(exc)[:300]})
            except Exception:
                pass
        finally:
            note_turn_end(t0)
            turn_task["task"] = None

            if sid and not closed["done"] and not was_cancelled:
                try:
                    asyncio.create_task(
                        _maybe_title(harness, str(sid), text, ws))
                except RuntimeError:
                    pass

    async def reader() -> None:
        """Read turn/confirm/cancel frames for the socket lifetime.

        ..
        """
        while not closed["done"]:
            try:
                msg = await ws.receive_json()
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "cancel":
                _cancel_turn()
                continue
            if msg.get("type") == "confirm":
                pending["ok"] = bool(msg.get("ok"))
                if pending["event"] is not None:
                    pending["event"].set()
                continue
            if msg.get("type") != "turn":
                continue
            if turn_task["task"] is not None:
                try:
                    await ws.send_json({"event": "error",
                                        "message": "turn already running"})
                except Exception:
                    pass
                continue
            text = str(msg.get("text", "") or "")
            if not text.strip():
                try:
                    await ws.send_json({"event": "error",
                                        "message": "empty text"})
                except Exception:
                    pass
                continue
            if msg.get("session_id"):
                _seen_sids.add(str(msg.get("session_id")))
            turn_task["task"] = asyncio.create_task(
                run_one(text, msg.get("session_id"), msg.get("cwd") or "."))

    await reader()
    closed["done"] = True
    if turn_task["task"] is not None:
        turn_task["task"].cancel()


@router.websocket("/deep")
async def deep(ws: WebSocket) -> None:
    """Autonomous turns over the same agent, cancellable.

    ..
    """
    await ws.accept()
    harness = get_agent()
    turn_task = {"task": None}
    closed = {"done": False}

    def _cancel_turn() -> bool:
        """Cancel the running autonomous turn.

        ..
        """
        task = turn_task["task"]
        if task is None:
            return False
        try:
            task.cancel()
        except Exception:
            pass
        return True

    async def run_one(text, sid, cwd) -> None:
        """Run one autonomous turn, streaming events to the socket.

        ..
        """
        t0 = note_turn_start()
        try:

            async for event in harness.run_text(text, session_id=sid, cwd=cwd,
                                                confirm_fn=lambda action: True):
                try:
                    await ws.send_json(_event_json(event))
                except Exception:
                    break
        except asyncio.CancelledError:
            await _send_cancelled(ws)
        except Exception as exc:
            try:
                await ws.send_json({"event": "error", "message": str(exc)[:300]})
            except Exception:
                pass
        finally:
            note_turn_end(t0)
            turn_task["task"] = None

    while not closed["done"]:
        try:
            msg = await ws.receive_json()
        except (WebSocketDisconnect, RuntimeError):
            break
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "cancel":
            _cancel_turn()
            continue
        if msg.get("type") != "turn":
            continue
        if turn_task["task"] is not None:
            try:
                await ws.send_json({"event": "error", "message": "turn already running"})
            except Exception:
                pass
            continue
        text = str(msg.get("text", "") or "")
        if not text.strip():
            try:
                await ws.send_json({"event": "error", "message": "empty text"})
            except Exception:
                pass
            continue
        if msg.get("session_id"):
            _seen_sids.add(str(msg.get("session_id")))
        turn_task["task"] = asyncio.create_task(run_one(text, msg.get("session_id"), msg.get("cwd") or "."))

    closed["done"] = True
    if turn_task["task"] is not None:
        turn_task["task"].cancel()
