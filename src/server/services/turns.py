"""Turn-socket helpers: event bridging and cancellation notices.

Contract: small transport helpers shared by both turn endpoints.
Dependencies: src.server.state for wav encoding, FastAPI WebSocket type.
Used by: src.server.routes.turns.
"""

from fastapi import WebSocket

from ..state import _wav_b64


def event_json(event) -> dict:
    """Bridge an agent event to WS JSON, extracting wav bytes to b64.

    ..
    """
    d = dict(event.data or {})
    wav = d.pop("wav", None)
    if wav is not None:
        d["wav_b64"] = _wav_b64(wav)
    return {"event": event.kind, **d}


async def send_cancelled(ws: WebSocket) -> None:
    """Best-effort cancelled notice. Never raises.

    ..
    """
    try:
        await ws.send_json({"event": "cancelled"})
    except Exception:
        pass
