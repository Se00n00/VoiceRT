"""Session middleware: load prior turns and write the turn back once.

``wrap_model_call`` prepends the stored transcript to the request only;
graph state stays this-turn-only, so ``add_messages`` cannot duplicate
history. ``after_agent`` finalizes and persists the turn exactly once
and hands the before/after snapshots to the L2 consolidator.
"""

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from src.agent.long_running.messages import finalize_turn, lc_messages
from src.agent.memory import JsonSessionMemory

__all__ = ["SessionMiddleware"]


def _state_messages(state: Any) -> list:
    """Messages list off an AgentState dict or object."""
    if isinstance(state, dict):
        return list(state.get("messages", []) or [])
    return list(getattr(state, "messages", []) or [])


def _contents(messages: list) -> list:
    """Role/content dicts -> content strings for eviction comparison."""
    return [
        str(m.get("content", "") or "")
        for m in messages or []
        if isinstance(m, dict)
    ]


class SessionMiddleware(AgentMiddleware):
    """One session's load/save boundary around the graph. One job."""

    def __init__(
        self,
        sessions: JsonSessionMemory,
        session_id: str | None,
        turn_text: str,
        on_saved: Callable[[str, list, list], None] | None = None,
    ):
        """Bind the store and turn identity; snapshot L1 before the run."""
        self.sessions = sessions
        self.session_id = session_id
        self.turn_text = str(turn_text or "")
        self.on_saved = on_saved
        self._saved = False
        try:
            self._before = _contents(
                self.sessions.history(session_id) if session_id else []
            )
        except Exception:
            self._before = []

    def _request(self, request):
        """Prepend stored history to one model request. Never raises."""
        if not self.session_id:
            return request
        try:
            history = lc_messages(self.sessions.history(self.session_id))
        except Exception:
            history = []
        if not history:
            return request
        return request.override(messages=[*history, *list(request.messages)])

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))

    def _save(self, state: Any) -> None:
        """Finalize and persist this turn once. Never raises."""
        if not self.session_id or self._saved:
            return
        try:
            result = finalize_turn(_state_messages(state))
            if not (self.turn_text or result.reply):
                return
            self.sessions.remember_turn(
                self.session_id, self.turn_text, result.reply
            )
            self._saved = True
            if self.on_saved is not None:
                after = _contents(self.sessions.history(self.session_id))
                self.on_saved(self.session_id, self._before, after)
        except Exception:
            pass

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._save(state)
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._save(state)
        return None
