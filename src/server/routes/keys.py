"""OS-keychain API key endpoints (status booleans, never values)."""

from fastapi import APIRouter, Request

from ..services import keys as service

router = APIRouter(tags=["keys"])


@router.get("/keys")
def key_status() -> dict:
    """Per-provider stored flags + store availability. Never values."""
    return service.status()


@router.post("/keys")
async def key_store(request: Request) -> dict:
    """Store one provider key in the OS keychain. Never echoes it."""
    host = request.client.host if request.client else ""
    denied = service.loopback_denied(host)
    if denied is not None:
        return denied
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    return service.store(payload)


@router.delete("/keys/{provider}")
def key_delete(provider: str, request: Request) -> dict:
    """Remove one provider key from the OS keychain."""
    host = request.client.host if request.client else ""
    denied = service.loopback_denied(host)
    if denied is not None:
        return denied
    return service.delete(provider)
