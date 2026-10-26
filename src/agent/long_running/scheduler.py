"""Concurrency for turns: admission slots and per-session serialization.

``AdmissionQueue`` bounds concurrent voice turns across the process.
``TurnScheduler`` serializes turns per session and injects a queued
event when a second turn arrives while the first is running.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from src.agent.events import AgentEvent
from src.models.runtime import FIFOScheduler

__all__ = ["AdmissionQueue", "TurnScheduler"]


class AdmissionQueue:
    """Process-wide FIFO admission slots with a bounded wait."""

    def __init__(self, max_inflight: int, timeout_s: float):
        """Build the scheduler; a broken build disables admission."""
        self.timeout_s = float(timeout_s or 0)
        try:
            self._sched = FIFOScheduler(max_inflight)
        except Exception:
            self._sched = None

    async def admit(self):
        """Take a slot, timing out when the server is saturated."""
        if self._sched is None:
            return None
        ticket = await asyncio.to_thread(
            self._sched.acquire, True, self.timeout_s
        )
        if ticket is None:
            raise TimeoutError(
                f"server saturated ({self._sched.max_concurrency} in flight, "
                f"queue waited {self.timeout_s:.0f}s); retry"
            )
        return ticket

    def release(self, ticket) -> None:
        """Hand a slot back. Never raises."""
        try:
            if ticket is not None and self._sched is not None:
                self._sched.release(ticket)
        except Exception:
            pass


class TurnScheduler:
    """Serialize turns per session; queue contenders."""

    def __init__(self):
        """Start with no locks and no pending input."""
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, list[str]] = {}

    async def run(
        self,
        sid: str | None,
        text: str,
        one: Callable[[str], AsyncIterator[AgentEvent]],
    ) -> AsyncIterator[AgentEvent]:
        """Run one text through ``one``, serialized by session id.

        A contender is recorded and returns a ``queued`` event; the
        active turn drains pending input before releasing the lock.
        ..
        """
        if sid is None:
            async for ev in one(str(text)):
                yield ev
            return
        lock = self._locks.setdefault(sid, asyncio.Lock())
        if lock.locked():
            pend = self._pending.setdefault(sid, [])
            pend.append(str(text))
            yield AgentEvent(
                node="term",
                kind="queued",
                data={"position": len(pend), "session_id": sid},
            )
            return
        await lock.acquire()
        try:
            texts = self._pending.pop(sid, []) + [str(text)]
            while texts:
                async for ev in one(texts.pop(0)):
                    yield ev
                texts.extend(self._pending.pop(sid, []))
        finally:
            lock.release()
            if not self._pending.get(sid):
                self._locks.pop(sid, None)
                self._pending.pop(sid, None)
