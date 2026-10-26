"""System endpoints: liveness, model/brain legs and metrics."""

from fastapi import APIRouter

from ..services import system as service

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    """Liveness, uptime, warm state and missing legs."""
    return service.health()


@router.get("/model")
def model_info() -> dict:
    """Current model name plus every configured profile."""
    return service.model_info()


@router.post("/model/switch")
async def model_switch(payload: dict) -> dict:
    """Switch the active model profile."""
    return await service.switch_model_by_name(payload or {})


@router.get("/brain")
def brain_info() -> dict:
    """Current brain mode plus the available modes."""
    return service.brain_info()


@router.post("/brain/switch")
async def brain_switch(payload: dict) -> dict:
    """Switch the brain mode."""
    return await service.switch_brain(payload or {})


@router.get("/metrics")
async def metrics(sid: str = "") -> dict:
    """Full runtime metrics snapshot for one optional session."""
    return await service.metrics(sid)
