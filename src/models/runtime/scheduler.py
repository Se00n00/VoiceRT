"""FIFO admission scheduler with a max-concurrency cap.

Tickets preserve arrival order: a thread only proceeds when its ticket is
the one being served AND a concurrency slot is free, so admission is
first-in-first-out instead of thundering-herd.
"""
import collections
import threading
import time


class FIFOScheduler:
    """Fair semaphore with FIFO ordering and introspection."""

    def __init__(self, max_concurrency=1):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = int(max_concurrency)
        self._cond = threading.Condition()
        self._queue = collections.deque()  # waiting ticket ids
        self._running = set()  # admitted ticket ids
        self._next_ticket = 0
        self._admitted_total = 0

    def acquire(self, blocking=True, timeout=None):
        """Take the next ticket and wait until it is admitted.

        Returns the ticket id on success, or None on timeout / non-blocking
        failure. Every returned ticket must be paired with release().
        """
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._queue.append(ticket)
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                if self._queue and self._queue[0] == ticket and len(self._running) < self.max_concurrency:
                    self._queue.popleft()
                    self._running.add(ticket)
                    self._admitted_total += 1
                    return ticket
                if not blocking:
                    self._queue.remove(ticket)
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    try:
                        self._queue.remove(ticket)
                    except ValueError:
                        pass
                    return None
                self._cond.wait(timeout=remaining)

    def release(self, ticket):
        """Free the slot held by ticket and wake the next waiter."""
        with self._cond:
            self._running.discard(ticket)
            self._cond.notify_all()

    def slot(self, blocking=True, timeout=None):
        """Context manager for `with scheduler.slot(): ...`."""
        scheduler = self

        class _Slot:
            def __enter__(self):
                self.ticket = scheduler.acquire(blocking=blocking, timeout=timeout)
                if self.ticket is None:
                    raise TimeoutError("FIFOScheduler: no slot became available in time")
                return self.ticket

            def __exit__(self, exc_type, exc, tb):
                scheduler.release(self.ticket)
                return False

        return _Slot()

    @property
    def pending(self):
        """Number of tickets waiting for admission."""
        with self._cond:
            return len(self._queue)

    @property
    def running(self):
        """Number of currently admitted (executing) tickets."""
        with self._cond:
            return len(self._running)

    @property
    def admitted_total(self):
        """Total admissions since construction."""
        with self._cond:
            return self._admitted_total

    def stats(self):
        """Dict with concurrency, pending, running, and totals."""
        with self._cond:
            return {
                "max_concurrency": self.max_concurrency,
                "pending": len(self._queue),
                "running": len(self._running),
                "admitted_total": self._admitted_total,
            }
