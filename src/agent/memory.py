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

``JsonSessionMemory`` keeps the exact same semantics but persists every
session to ``<sessions_dir>/<sid>.json`` (atomic tmp+rename writes), so
conversation history survives restarts. This is the store ``VoiceAgent``
uses by default.
"""
import os
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

__all__ = ["LangChainSessionMemory", "LCSessionHistory", "JsonSessionMemory",
           "new_session_id"]


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


def _safe_sid(sid: str) -> str:
    """Filesystem-safe session id (anything else becomes a hex digest)."""
    import hashlib
    import re

    s = str(sid or "")
    if s and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", s):
        return s
    return "sid-" + hashlib.sha256(s.encode()).hexdigest()[:32]


class JsonSessionMemory(LangChainSessionMemory):
    """Same window/TTL/LRU semantics, persisted per session as JSON.

    Every mutation writes ``<sessions_dir>/<sid>.json`` atomically
    (tmp file + ``os.replace``); a sid missing from RAM is reloaded from
    disk on next access (TTL-checked). Stale files are unlinked
    opportunistically on the write path; LRU eviction unlinks too.
    """

    def __init__(self, sessions_dir: str = "sessions", **kw):
        super().__init__(**kw)
        self.sessions_dir = str(sessions_dir or "sessions")

    # -- persistence --------------------------------------------------
    def _path_locked(self, sid: str) -> str:
        return os.path.join(self.sessions_dir, _safe_sid(sid) + ".json")

    def _save_locked(self, sid: str) -> None:
        try:
            os.makedirs(self.sessions_dir, exist_ok=True)
            payload = {
                "session_id": str(sid),
                "at": time.time(),
                "messages": [_to_dict(m) for m in self._data[sid].messages],
            }
            import json as _json

            tmp = self._path_locked(sid) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
            os.replace(tmp, self._path_locked(sid))
        except Exception:
            pass
        self._sweep_files_locked(time.time())

    def _sweep_files_locked(self, now: float) -> None:
        """Unlink session files older than TTL. Best-effort, never raises."""
        try:
            cutoff = now - self.max_age_s
            for name in os.listdir(self.sessions_dir):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(self.sessions_dir, name)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.unlink(path)
                except Exception:
                    pass
        except Exception:
            pass

    def _load_locked(self, sid: str, now: float) -> "LCSessionHistory | None":
        try:
            import json as _json

            with open(self._path_locked(sid), "r", encoding="utf-8") as f:
                payload = _json.load(f)
            at = float(payload.get("at", 0.0) or 0.0)
            if now - at > self.max_age_s:
                try:
                    os.unlink(self._path_locked(sid))
                except Exception:
                    pass
                return None
            h = LCSessionHistory()
            for m in payload.get("messages", []) or []:
                if isinstance(m, dict) and m.get("content"):
                    h.messages.append(_to_pair(str(m.get("role", "user")),
                                              str(m.get("content"))))
            h.at = now
            self._prune_locked(h)
            return h
        except Exception:
            return None

    def _unlink_locked(self, sid: str) -> None:
        try:
            os.unlink(self._path_locked(sid))
        except Exception:
            pass

    # -- overrides ----------------------------------------------------
    def _get_locked(self, sid, now) -> LCSessionHistory:
        h = self._data.get(sid)
        if h is None:
            h = self._load_locked(sid, now)
            if h is None:
                if len(self._data) >= self.max_sessions:
                    oldest = min(self._data, key=lambda k: self._data[k].at)
                    self._unlink_locked(oldest)
                    del self._data[oldest]
                h = LCSessionHistory()
                self._seq += 1
            else:
                self._seq += 1
            self._data[sid] = h
        h.at = now
        return h

    def lc_add(self, sid, role, content) -> None:
        now = time.time()
        with self._lock:
            self._sweep_locked(now)
            h = self._get_locked(sid, now)
            if not content:
                return
            h.messages.append(_to_pair(role, content))
            self._prune_locked(h)
            self._save_locked(sid)

    def reset(self, sid) -> None:
        now = time.time()
        with self._lock:
            h = self._data.get(sid)
            if h is not None:
                h.messages = []
                h.at = now
                self._save_locked(sid)
            else:
                self._unlink_locked(sid)

    def drop(self, sid) -> bool:
        with self._lock:
            gone = self._data.pop(sid, None) is not None
            self._unlink_locked(sid)
            return gone

    def stats(self) -> dict:
        with self._lock:
            return {
                "sessions": len(self._data),
                "turns": sum(len(h.messages) for h in self._data.values()),
                "max_turns": self.max_turns,
                "max_age_s": self.max_age_s,
                "backend": "json",
                "dir": self.sessions_dir,
            }
