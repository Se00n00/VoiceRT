"""Liveness + model-leg routes."""

import time

from fastapi import APIRouter

from .. import state
from ..state import (
    MODEL_PROFILES,
    _current_model,
    _t_boot,
    switch_model,
)

router = APIRouter()


@router.get("/health")
def health():
    try:
        loaded = bool(state._agent is not None and getattr(state._agent, "warmed", False))
    except Exception:
        loaded = False
    try:
        missing = list(getattr(state._agent, "missing", []) or [])
    except Exception:
        missing = []
    return {"ok": True, "uptime_s": time.time() - _t_boot,
            "agent_loaded": loaded, "missing": missing}


@router.get("/model")
def model_info():
    cur = _current_model.get("name")
    avail = []
    for key, p in MODEL_PROFILES.items():
        avail.append({"name": key, "label": p["label"], "backend": p["backend"],
                      "desc": p["desc"], "ready": bool(p.get("ready")),
                      "current": key == cur})
    return {"current": cur, "available": avail}


@router.post("/model/switch")
async def model_switch(payload: dict):
    return await switch_model((payload or {}).get("name", ""))
