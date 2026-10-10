"""Domain routers for the voice-term bridge (mounted by ``src.server``)."""

from .chat import router as chat_router
from .console import router as console_router
from .sessions import router as sessions_router
from .system import router as system_router
from .talk import router as talk_router
from .turns import router as turns_router
from .voice import router as voice_router

__all__ = [
    "chat_router",
    "console_router",
    "sessions_router",
    "system_router",
    "talk_router",
    "turns_router",
    "voice_router",
]
