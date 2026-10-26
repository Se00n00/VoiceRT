"""Keychain service: provider key status, storage and deletion.

Contract: plain dict results; never returns or logs secret values.
Dependencies: src.agent.keystore, src.models.provider (lazy).
Used by: src.server.routes.keys.
"""

from src.agent.keystore import (
    delete_secret,
    has_secret,
    set_secret,
    store_available,
    store_name,
)


def loopback_denied(host: str) -> dict | None:
    """Error body when the peer is not loopback, else None.

    Key writes mutate credentials, so they stay on this machine even
    though the bridge's CORS policy is permissive.
    ..
    """
    if str(host or "") in ("127.0.0.1", "::1"):
        return None
    return {"kind": "error", "message": "keys API is loopback-only"}


def status() -> dict:
    """Per-provider stored flags plus store availability.

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


def store(payload: dict) -> dict:
    """Store one provider key in the OS keychain without echoing it.

    ..
    """
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
    except Exception as exc:
        return {"kind": "error",
                "message": f"store failed: {type(exc).__name__}"[:300]}
    return {"kind": "ok", "provider": provider, "stored": True}


def delete(provider: str) -> dict:
    """Remove one provider key from the OS keychain.

    ..
    """
    name = str(provider or "").strip().lower()
    from src.models.provider import ChatModel

    if name not in ChatModel.BACKENDS:
        return {"kind": "error",
                "message": f"unknown provider {name!r}; "
                           f"available: {list(ChatModel.BACKENDS)}"}
    return {"kind": "ok", "provider": name,
            "cleared": delete_secret(name)}
