"""Domain routers for the voice-term bridge (mounted by ``src.server``)."""

from .sessions import router as sessions_router
from .system import router as system_router
from .turns import router as turns_router
from .voice import router as voice_router

__all__ = ["sessions_router", "system_router", "turns_router", "voice_router"]
