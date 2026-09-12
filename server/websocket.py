"""Bidirectional talk WebSocket: wav bytes in -> JSON turn out.

Protocol (mirrors POST /v1/voice but streaming over one socket):
- client -> server: binary wav bytes (one utterance per message)
- server -> client: JSON text frame {"text", "reply", "wav_b64",
  "ttfa_s", "total_s", "vram_mb"} per utterance
- client may send {"type": "close"} text or just disconnect to end.

The engine is imported lazily per connection so importing this module
never pulls torch/weights.
"""
import asyncio
import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


def _get_engine():
    # Share the HTTP singleton: a second VoiceEngine would duplicate all
    # four models in VRAM (~2GB) and OOM the 4GB card.
    from server.routes import get_engine

    return get_engine()


# One shared instance per process once first WS connects.
_engine = None


def get_ws_engine():
    global _engine
    if _engine is None:
        _engine = _get_engine()
    return _engine


def _decode_wav_bytes(data: bytes):
    from server.routes import _load_wav_sync

    return _load_wav_sync(data)


@router.websocket("/v1/talk")
async def talk(ws: WebSocket):
    await ws.accept()
    try:
        eng = get_ws_engine()
    except Exception as exc:  # weights/torch missing: report, don't hang
        await ws.send_json({"ok": False, "error": f"engine unavailable: {exc}"})
        await ws.close()
        return
    # Optional ?session_id= query param; auto-issued otherwise and echoed
    # back in every turn frame so the frontend can persist it.
    from engine.session import new_session_id

    sid = ws.query_params.get("session_id") or new_session_id()
    while True:
        try:
            msg = await ws.receive()
        except WebSocketDisconnect:
            break
        data = msg.get("bytes")
        if data is None:
            # Ignore/handle text frames: "close" ends the loop.
            if msg.get("text") == "close":
                break
            continue
        try:
            audio, sr = await asyncio.to_thread(_decode_wav_bytes, bytes(data))
            from server.routes import _guard_wav

            _guard_wav(audio, sr)
            r = await asyncio.to_thread(eng.stream_turn, audio, sr, sid)
            from runtime.tensor import to_host_numpy

            wav = to_host_numpy(r["wav"])
            await ws.send_json(
                {
                    "text": r["text"],
                    "reply": r["reply"],
                    "wav_b64": base64.b64encode(wav.tobytes()).decode(),
                    "ttfa_s": r["ttfa_s"],
                    "total_s": r["total_s"],
                    "vram_mb": r.get("vram_mb"),
                    "session_id": sid,
                }
            )
        except WebSocketDisconnect:
            break
        except Exception as exc:
            try:
                await ws.send_json({"ok": False, "error": str(exc)[:300]})
            except Exception:
                break
