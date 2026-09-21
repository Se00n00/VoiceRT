"""Per-session memory as a LangChain abstraction.

``LangChainSessionMemory`` is the session store for the voice agent, built
on ``langchain_core`` message objects instead of raw dicts. Same surface as
the legacy ``engine.SessionStore`` (``history`` / ``remember_turn`` /
``reset`` / ``drop`` / ``stats``) so ``VoiceAgent`` and ``server.py`` swap
over with no behavior change:

- one ``LCBackend`` (a ``BaseChatMessageHistory``) per ``session_id``
- window cap (``max_turns`` → message cap, oldest pruned)
- TTL + LRU cap on sessions (RAM-only, no persistence)
- ``history(sid)`` still returns ``[{role, content}]`` dicts for the LLM
  chat-template path; ``lc_history(sid)`` exposes LangChain messages for
  LangChain chains (``RunnableWithMessageHistory``-style consumers).
"""
import threading
import time

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    _HAVE_LC = True
except Exception:  # pragma: no cover - langchain absent (never in prod)
    _HAVE_LC = False
    BaseChatMessageHistory = object  # type: ignore

    class _Msg:  # minimal Human/AI stand-ins (dict-backed, test-safe)
        def __init__(self, content=""):
            self.content = str(content)
            self.type = "human" if type(self).__name__ == "HumanMessage" else "ai"

    class HumanMessage(_Msg):  # type: ignore
        pass

    class AIMessage(_Msg):  # type: ignore
        pass

__all__ = ["LangChainSessionMemory", "LCSessionHistory", "new_session_id"]


def new_session_id():
    from engine import new_session_id as _new

    return _new()


class LCSessionHistory(BaseChatMessageHistory):
    """One session's LangChain message history (in-RAM list)."""

    def __init__(self):
        self.messages: list = []
        self.at: float = time.time()

    def add_messages(self, messages) -> None:
        self.messages.extend(messages)
        self.at = time.time()

    def clear(self) -> None:
        self.messages = []
        self.at = time.time()


def _to_pair(role: str, content: str):
    if role == "assistant":
        return AIMessage(content=str(content))
    return HumanMessage(content=str(content))


def _to_dict(msg) -> dict:
    try:
        kind = msg.type  # "human" | "ai" (+ system/tool, mapped below)
    except Exception:
        kind = "human"
    role = {"human": "user", "ai": "assistant"}.get(kind, kind)
    try:
        content = str(msg.content)
    except Exception:
        content = ""
    return {"role": role, "content": content}


class LangChainSessionMemory:
    """``session_id -> LCSessionHistory`` with window/TTL/LRU caps.

    Drop-in replacement for ``engine.SessionStore``.
    """

    def __init__(self, max_turns=20, max_age_s=1800, max_sessions=1000):
        self.max_turns = int(max_turns)
        self.max_age_s = float(max_age_s)
        self.max_sessions = int(max_sessions)
        self._lock = threading.Lock()
        self._data: dict[str, LCSessionHistory] = {}
        self._seq = 0

    # -- internals ------------------------------------------------------
    def _sweep_locked(self, now):
        cutoff = now - self.max_age_s
        for sid in [s for s, h in self._data.items() if h.at < cutoff]:
            del self._data[sid]

    def _get_locked(self, sid, now) -> LCSessionHistory:
        h = self._data.get(sid)
        if h is None:
            if len(self._data) >= self.max_sessions:
                oldest = min(self._data, key=lambda k: self._data[k].at)
                del self._data[oldest]
            h = LCSessionHistory()
            self._data[sid] = h
            self._seq += 1
        h.at = now
        return h

    def _prune_locked(self, h: LCSessionHistory):
        cap = max(1, self.max_turns) * 2  # user+assistant per turn
        if len(h.messages) > cap:
            del h.messages[:len(h.messages) - cap]

    # -- LangChain surface ----------------------------------------------
    def lc_history(self, sid) -> LCSessionHistory:
        """The LangChain history object (for chains / RunnableWithMessageHistory)."""
        now = time.time()
        with self._lock:
            self._sweep_locked(now)
            return self._get_locked(sid, now)

    def lc_add(self, sid, role, content) -> None:
        if not content:
            return
        now = time.time()
        with self._lock:
            self._sweep_locked(now)
            h = self._get_locked(sid, now)
            h.messages.append(_to_pair(role, content))
            self._prune_locked(h)

    # -- legacy dict surface (VoiceAgent / server.py use this) ----------
    def history(self, sid) -> list:
        return [_to_dict(m) for m in self.lc_history(sid).messages]

    def append(self, sid, role, content) -> None:
        self.lc_add(sid, role, content)

    def remember_turn(self, sid, user_text, assistant_text) -> None:
        self.append(sid, "user", user_text)
        self.append(sid, "assistant", assistant_text)

    def reset(self, sid) -> None:
        now = time.time()
        with self._lock:
            h = self._data.get(sid)
            if h is not None:
                h.messages = []
                h.at = now

    def drop(self, sid) -> bool:
        with self._lock:
            return self._data.pop(sid, None) is not None

    def stats(self) -> dict:
        with self._lock:
            return {
                "sessions": len(self._data),
                "turns": sum(len(h.messages) for h in self._data.values()),
                "max_turns": self.max_turns,
                "max_age_s": self.max_age_s,
                "backend": "langchain",
            }
