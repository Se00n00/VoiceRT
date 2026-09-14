"""Latency profiler: time stages with a context manager, then summarize."""
import contextlib
import statistics
import threading
import time


class Profiler:
    """Thread-safe recorder of named latency samples."""

    def __init__(self):
        self._lock = threading.Lock()
        self._samples = {}  # name -> list[seconds]

    def record(self, name, seconds):
        """Append one latency sample (seconds) for a stage."""
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError("latency samples must be >= 0")
        with self._lock:
            self._samples.setdefault(str(name), []).append(seconds)

    @contextlib.contextmanager
    def time(self, name):
        """Context manager: `with prof.time('llm'): ...` records the span."""
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self.record(name, time.perf_counter() - t0)

    def timed(self, name, fn, *args, **kwargs):
        """Call fn(*args, **kwargs), record its latency, return its result."""
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.record(name, time.perf_counter() - t0)

    def counts(self):
        """Number of samples per stage."""
        with self._lock:
            return {k: len(v) for k, v in self._samples.items()}

    def summary(self):
        """Per-stage {count, total_s, mean_ms, min_ms, max_ms, p50_ms}."""
        with self._lock:
            snapshot = {k: list(v) for k, v in self._samples.items()}
        out = {}
        for name, vals in snapshot.items():
            if not vals:
                continue
            ordered = sorted(vals)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                p50 = ordered[mid]
            else:
                p50 = (ordered[mid - 1] + ordered[mid]) / 2.0
            out[name] = {
                "count": len(vals),
                "total_s": float(sum(vals)),
                "mean_ms": float(statistics.fmean(vals)) * 1000.0,
                "min_ms": float(min(vals)) * 1000.0,
                "max_ms": float(max(vals)) * 1000.0,
                "p50_ms": float(p50) * 1000.0,
            }
        return out

    def report(self):
        """One human-readable line per stage."""
        lines = []
        for name, s in sorted(self.summary().items()):
            lines.append(
                f"{name}: n={s['count']} mean={s['mean_ms']:.1f}ms "
                f"p50={s['p50_ms']:.1f}ms "
                f"min={s['min_ms']:.1f}ms max={s['max_ms']:.1f}ms"
            )
        return "\n".join(lines)

    def reset(self):
        """Drop all recorded samples."""
        with self._lock:
            self._samples.clear()
