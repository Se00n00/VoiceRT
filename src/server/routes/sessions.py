"""Persisted-session reads (no model calls)."""

from fastapi import APIRouter

from ..state import get_agent

router = APIRouter()


@router.get("/term/title")
def term_title(sid: str = ""):
    """Persisted name for a session, so a resume keeps it.

    Read-only and model-free on purpose: the TUI calls this once on
    mount, and ``voicert -s <id>`` followed by an immediate quit never
    runs a turn, so there is nothing to generate a title from.
    """
    sid = str(sid or "").strip()
    if not sid:
        return {"title": "", "session_id": ""}
    try:
        title = get_agent().sessions.get_title(sid)
    except Exception:
        title = ""
    return {"title": title or "", "session_id": sid}


@router.get("/term/history")
def term_history(sid: str = ""):
    """Full persisted transcript for a session (past conversations).

    Reads ``sessions/<sid>.json`` straight off disk — no model, no
    TTL prune — so the UI can restore user/assistant turns plus the
    richer ``thinking`` / ``tool`` / ``cot`` roles the agent never
    feeds back into the LLM window (see ``_lc_messages``: user and
    assistant only). Touches the file so the session sweeper does
    not reap it while it is being viewed.
    """
    import json as _json
    import os as _os

    from src.agent.memory import _safe_sid

    sid = str(sid or "").strip()
    if not sid:
        return {"session_id": "", "title": "", "messages": []}
    try:
        base = getattr(get_agent().sessions, "sessions_dir", "sessions") or "sessions"
        path = _os.path.join(str(base), _safe_sid(sid) + ".json")
        with open(path, "r", encoding="utf-8") as f:
            payload = _json.load(f)
        try:
            _os.utime(path, None)
        except Exception:
            pass
        msgs = payload.get("messages", []) or []
        msgs = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in msgs
            if isinstance(m, dict) and str(m.get("content", "") or "").strip()
        ]
        return {
            "session_id": str(payload.get("session_id", sid) or sid),
            "title": str(payload.get("title", "") or ""),
            "messages": msgs,
        }
    except Exception:
        return {"session_id": sid, "title": "", "messages": []}
