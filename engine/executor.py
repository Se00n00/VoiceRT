"""Per-leg ThreadPool executor: one pool per pipeline leg."""
import concurrent.futures
import threading

from engine.model import LEG_ORDER


class LegExecutor:
    """Owns a ThreadPoolExecutor per leg so slow legs never starve others."""

    def __init__(self, workers_per_leg=1, default_workers=1):
        if isinstance(workers_per_leg, int):
            per_leg = {leg: workers_per_leg for leg in LEG_ORDER}
        else:
            per_leg = {leg: int(workers_per_leg.get(leg, default_workers)) for leg in LEG_ORDER}
        self._pools = {
            leg: concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, n), thread_name_prefix=f"leg-{leg}"
            )
            for leg, n in per_leg.items()
        }
        self._lock = threading.Lock()
        self._shutdown = False

    def _pool(self, leg):
        with self._lock:
            if self._shutdown:
                raise RuntimeError("LegExecutor is shut down")
            try:
                return self._pools[leg]
            except KeyError:
                raise KeyError(f"unknown leg {leg!r}; expected one of {sorted(self._pools)}")

    def submit(self, leg, fn, *args, **kwargs):
        """Submit fn(*args, **kwargs) to the leg's pool; returns a Future."""
        return self._pool(leg).submit(fn, *args, **kwargs)

    def wait(self, future, timeout=None):
        """Block for a Future from submit() and return (or raise) its result."""
        return future.result(timeout=timeout)

    def run(self, leg, fn, *args, timeout=None, **kwargs):
        """Submit to a leg and wait; convenience for one-shot calls."""
        return self.wait(self.submit(leg, fn, *args, **kwargs), timeout=timeout)

    def map(self, leg, fn, items, timeout=None):
        """Apply fn to each item on the leg's pool, preserving order."""
        futs = [self.submit(leg, fn, item) for item in items]
        return [f.result(timeout=timeout) for f in futs]

    def legs(self):
        """Leg names with pools."""
        with self._lock:
            return sorted(self._pools)

    def shutdown(self, wait=True, cancel_futures=False):
        """Shut down every pool; further submit() calls raise."""
        with self._lock:
            self._shutdown = True
            pools = list(self._pools.values())
        for pool in pools:
            try:
                pool.shutdown(wait=wait, cancel_futures=cancel_futures)
            except TypeError:
                # Older Pythons lack cancel_futures.
                pool.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(wait=True)
        return False
