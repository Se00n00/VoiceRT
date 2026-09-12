"""Per-session conversation history (multi-turn memory).

The frontend owns the session id (a UUID it generates once and sends with
every call). The server keeps the last N turns per session in memory with
a TTL, so follow-up prompts ("and the second one?") resolve against what
was already said. Stateless when the client sends no id.

Bounds (production): max_turns per session (oldest pruned), max_age_s TTL
with lazy sweep on access, max_sessions cap (oldest-access evicted).
All methods are thread-safe (uvicorn workers share one store).
"""
import threading
import time
import uuid

__all__ = ["SessionStore", "new_session_id"]


def new_session_id():
    return uuid.uuid4().hex


class SessionStore:
    """session_id -> [{role, content}] with caps. Real code, no backend."""

    def __init__(self, max_turns=20, max_age_s=1800, max_sessions=1000):
        self.max_turns = int(max_turns)
        self.max_age_s = float(max_age_s)
        self.max_sessions = int(max_sessions)
        self._lock = threading.Lock()
        self._data = {}   # sid -> {"msgs": [...], "at": last_access}
        self._seq = 0

    # -- internals ------------------------------------------------------
    def _sweep(self, now):
        """Drop expired sessions. Caller must hold the lock."""
        cutoff = now - self.max_age_s
        dead = [sid for sid, s in self._data.items() if s["at"] < cutoff]
        for sid in dead:
            del self._data[sid]

    def _get_locked(self, sid, now):
        s = self._data.get(sid)
        if s is None:
            if len(self._data) >= self.max_sessions:
                # evict least-recently-used session
                oldest = min(self._data, key=lambda k: self._data[k]["at"])
                del self._data[oldest]
            s = {"msgs": [], "at": now, "seq": self._seq}
            self._seq += 1
            self._data[sid] = s
        s["at"] = now
        return s

    # -- public ----------------------------------------------------------
    def history(self, sid):
        """Copy of stored messages (oldest first). Auto-creates session."""
        now = time.time()
        with self._lock:
            self._sweep(now)
            return list(self._get_locked(sid, now)["msgs"])

    def append(self, sid, role, content):
        """Append one message; prune oldest turns past max_turns."""
        if not content:
            return
        now = time.time()
        with self._lock:
            self._sweep(now)
            s = self._get_locked(sid, now)
            s["msgs"].append({"role": role, "content": content})
            if len(s["msgs"]) > self.max_turns:
                del s["msgs"][:len(s["msgs"]) - self.max_turns]

    def remember_turn(self, sid, user_text, assistant_text):
        self.append(sid, "user", user_text)
        self.append(sid, "assistant", assistant_text)

    def reset(self, sid):
        with self._lock:
            if sid in self._data:
                self._data[sid]["msgs"] = []
                self._data[sid]["at"] = time.time()

    def drop(self, sid):
        with self._lock:
            return self._data.pop(sid, None) is not None

    def stats(self):
        with self._lock:
            return {
                "sessions": len(self._data),
                "turns": sum(len(s["msgs"]) for s in self._data.values()),
                "max_turns": self.max_turns,
                "max_age_s": self.max_age_s,
            }
