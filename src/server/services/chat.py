"""Chat service: shared session-history window for Qwen chat.

Contract: history_messages(agent, sid) -> list of user/assistant dicts.
Dependencies: only the agent session store.
Used by: src.server.routes.chat and src.server.services.talk.
"""

_HISTORY_TURNS = 20


def history_messages(agent, sid: str) -> list:
    """Last user/assistant turns for a session, oldest first.

    ..
    """
    try:
        sessions = getattr(agent, "sessions", None)
        if sessions is None or not sid:
            return []
        hist = sessions.history(sid) or []
        msgs = [{"role": str(m.get("role", "user") or "user"),
                 "content": str(m.get("content", "") or "")}
                for m in hist if isinstance(m, dict)
                and str(m.get("content", "") or "").strip()
                and str(m.get("role", "user") or "user") in ("user", "assistant")]
        return msgs[-(_HISTORY_TURNS * 2):]
    except Exception:
        return []
