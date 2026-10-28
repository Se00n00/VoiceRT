"""Session service: persisted titles and transcripts without model calls.

Contract: plain dict results keyed for the session routes.
Dependencies: src.agent.memory, the agent's session directory.
Used by: src.server.routes.sessions.
"""

import json
import os

from src.agent.memory import _safe_sid
from src.paths import VoicertPaths

from ..state import get_agent


def title(sid: str) -> dict:
    """Persisted name for a session, so a resume keeps it.

    ..
    """
    sid = str(sid or "").strip()
    if not sid:
        return {"title": "", "session_id": ""}
    try:
        name = get_agent().sessions.get_title(sid)
    except Exception:
        name = ""
    return {"title": name or "", "session_id": sid}


def history(sid: str) -> dict:
    """Full persisted transcript for a session, straight off disk.

    ..
    """
    sid = str(sid or "").strip()
    if not sid:
        return {"session_id": "", "title": "", "messages": []}
    try:
        base = getattr(
            get_agent().sessions, "sessions_dir", VoicertPaths.sessions_dir()
        ) or VoicertPaths.sessions_dir()
        path = os.path.join(str(base), _safe_sid(sid) + ".json")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        try:
            os.utime(path, None)
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
