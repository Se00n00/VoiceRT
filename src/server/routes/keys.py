"""OS-keychain API key management (status booleans, never values)."""

from fastapi import APIRouter, Request

from src.agent.keystore import (
    delete_secret,
    has_secret,
    set_secret,
    store_available,
    store_name,
)

router = APIRouter()


def _loopback_only(request: Request) -> dict | None:
    """Error body when the peer is not loopback, else None.

    Key writes mutate credentials, so they stay on this machine even
    though the bridge's CORS policy is permissive.
    ..
    """
    try:
        host = str((request.client.host if request.client else "") or "")
    except Exception:
        host = ""
    if host in ("127.0.0.1", "::1"):
        return None
    return {"kind": "error", "message": "keys API is loopback-only"}


@router.get("/keys")
def key_status() -> dict:
    """Per-provider stored flags + store availability. Never values.

    ..
    """
    try:
        from src.models.provider import ChatModel

        providers = {b: has_secret(b) for b in ChatModel.BACKENDS}
    except Exception:
        from src.models.provider import ChatModel

        providers = {b: False for b in ChatModel.BACKENDS}
    return {"providers": providers, "store": store_name(),
            "available": store_available()}


@router.post("/keys")
async def key_store(request: Request) -> dict:
    """Store one provider key in the OS keychain. Never echoes it.

    ..
    """
    denied = _loopback_only(request)
    if denied is not None:
        return denied
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    provider = str((payload or {}).get("provider", "") or "").strip().lower()
    secret = str((payload or {}).get("key", "") or "")
    from src.models.provider import ChatModel

    if provider not in ChatModel.BACKENDS:
        return {"kind": "error",
                "message": f"unknown provider {provider!r}; "
                           f"available: {list(ChatModel.BACKENDS)}"}
    try:
        set_secret(provider, secret)
    except (ValueError, RuntimeError) as exc:
        return {"kind": "error", "message": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001 - surfaced, never raised
        return {"kind": "error",
                "message": f"store failed: {type(exc).__name__}"[:300]}
    return {"kind": "ok", "provider": provider, "stored": True}


@router.delete("/keys/{provider}")
def key_delete(provider: str, request: Request) -> dict:
    """Remove one provider key from the OS keychain.

    ..
    """
    denied = _loopback_only(request)
    if denied is not None:
        return denied
    name = str(provider or "").strip().lower()
    from src.models.provider import ChatModel

    if name not in ChatModel.BACKENDS:
        return {"kind": "error",
                "message": f"unknown provider {name!r}; "
                           f"available: {list(CLOUD_BACKENDS)}"}
    return {"kind": "ok", "provider": name,
            "cleared": delete_secret(name)}
