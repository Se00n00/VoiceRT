"""Pure Qwen chat socket (no tools, no confirms, no delegate).

Protocol::

    client -> server: {"type": "chat", "text": ..., "session_id": ...}
    server -> client: {"event": "token", "piece": ...}   (chunked reply)
    server -> client: {"event": "reply", "reply": ...}
    server -> client: {"event": "error", "message": ...}

Only serves while the qwen profile is active — anything else errors out
instead of double-loading weights next to the voice legs. Turns are
appended to the shared session store (last 20 kept as context).
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..state import console_emit, get_agent, note_turn_end, note_turn_start, qwen_leg

router = APIRouter()

_HISTORY_TURNS = 20


def _history_messages(agent, sid: str) -> list:
    try:
        sessions = getattr(agent, "sessions", None)
        if sessions is None or not sid:
            return []
        hist = sessions.history(sid) or []
        msgs = [{"role": str(m.get("role", "user") or "user"),
                 "content": str(m.get("content", "") or "")}
                for m in hist if isinstance(m, dict)
                and str(m.get("content", "") or "").strip()
                and str(m.get("role", "user") or "user") in ("user", "assistant")]
        return msgs[-(_HISTORY_TURNS * 2):]
    except Exception:
        return []


@router.websocket("/chat")
async def chat(ws: WebSocket):
    await ws.accept()
    closed = {"done": False}
    try:
        while not closed["done"]:
            try:
                msg = await ws.receive_json()
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                continue
            if not isinstance(msg, dict) or msg.get("type") != "chat":
                continue
            text = str(msg.get("text", "") or "")
            if not text.strip():
                try:
                    await ws.send_json({"event": "error", "message": "empty text"})
                except Exception:
                    pass
                continue
            sid = str(msg.get("session_id", "") or "")
            t0 = note_turn_start()
            try:
                agent = get_agent()
                llm = qwen_leg(agent)
                if llm is None:
                    raise RuntimeError("no qwen leg available")
                console_emit("chat", "user", text, sid)
                messages = _history_messages(agent, sid) + [
                    {"role": "user", "content": text}]
                try:
                    res = await llm.generate(messages, max_tokens=512)
                    reply = str(getattr(res, "text", "") or "").strip()
                except Exception as exc:
                    raise RuntimeError(f"generate failed: {exc}") from exc
                if not reply:
                    raise RuntimeError("empty reply")
                words = reply.split(" ")
                step = max(1, len(words) // 6)
                for i in range(0, len(words), step):
                    try:
                        await ws.send_json({"event": "token",
                                            "piece": " ".join(words[i:i + step]) + " "})
                    except Exception:
                        break
                    await asyncio.sleep(0)
                try:
                    sessions = getattr(agent, "sessions", None)
                    if sessions is not None and sid:
                        sessions.append(sid, "user", text)
                        sessions.append(sid, "assistant", reply)
                except Exception:
                    pass
                try:
                    await ws.send_json({"event": "reply", "reply": reply})
                    console_emit("chat", "agent", reply, sid)
                except Exception:
                    pass
            except Exception as exc:
                console_emit("chat", "error", str(exc)[:300], sid)
                try:
                    await ws.send_json({"event": "error", "message": str(exc)[:300]})
                except Exception:
                    pass
            finally:
                note_turn_end(t0)
    finally:
        closed["done"] = True
