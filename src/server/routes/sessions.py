"""Persisted-session read endpoints (no model calls)."""

from fastapi import APIRouter

from ..services import sessions as service

router = APIRouter(tags=["sessions"])


@router.get("/term/title")
def term_title(sid: str = "") -> dict:
    """Persisted name for a session, so a resume keeps it."""
    return service.title(sid)


@router.get("/term/history")
def term_history(sid: str = "") -> dict:
    """Full persisted transcript for a session (past conversations)."""
    return service.history(sid)
