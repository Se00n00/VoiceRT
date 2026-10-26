"""WebRTC duplex-call signaling endpoint."""

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from ..services import talk as service

router = APIRouter(tags=["talk"])


@router.post("/talk/offer")
async def talk_offer(payload: dict) -> Response:
    """WebRTC offer/answer handshake for a duplex voice call."""
    status, body = await service.offer(payload or {})
    return JSONResponse(status_code=status, content=body)
