"""Timing events with CUDA-event precision and perf_counter fallback."""
import time

import torch


class TimerEvent:
    """Start/stop timestamp usable on CUDA and CPU.

    On CUDA (and enable_timing=True) this wraps torch.cuda.Event for
    GPU-side timing; everywhere else it records time.perf_counter().
    elapsed_ms() works for any pair recorded the same way.
    """

    def __init__(self, enable_timing=True):
        self.enable_timing = bool(enable_timing)
        self._cuda_event = None
        self._cpu_time = None

    def record(self, stream=None):
        """Stamp the current time; returns self for chaining."""
        if (
            self.enable_timing
            and torch.cuda.is_available()
        ):
            try:
                ev = torch.cuda.Event(enable_timing=True)
                if stream is not None:
                    inner = getattr(stream, "stream", stream)
                    if inner is not None:
                        ev.record(inner)
                    else:
                        ev.record()
                else:
                    ev.record()
                self._cuda_event = ev
                self._cpu_time = None
                return self
            except Exception:
                pass
        self._cuda_event = None
        self._cpu_time = time.perf_counter()
        return self

    def elapsed_ms(self, end):
        """Milliseconds from this event to `end` (float, >= 0)."""
        if not isinstance(end, TimerEvent):
            raise TypeError(f"elapsed_ms expects a TimerEvent, got {type(end)}")
        if self._cuda_event is not None and end._cuda_event is not None:
            try:
                return max(0.0, float(self._cuda_event.elapsed_time(end._cuda_event)))
            except Exception:
                pass
        if self._cpu_time is not None and end._cpu_time is not None:
            return max(0.0, (end._cpu_time - self._cpu_time) * 1000.0)
        # Mixed CUDA/CPU stamps: synchronize then compare against now.
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        now = time.perf_counter()
        start = self._cpu_time if self._cpu_time is not None else now
        stop = end._cpu_time if end._cpu_time is not None else now
        return max(0.0, (stop - start) * 1000.0)


def now_event(stream=None):
    """Record and return a fresh TimerEvent."""
    return TimerEvent().record(stream)


def elapsed_ms(start, end):
    """Milliseconds between two TimerEvents."""
    if not isinstance(start, TimerEvent) or not isinstance(end, TimerEvent):
        raise TypeError("elapsed_ms expects two TimerEvents")
    return start.elapsed_ms(end)


def time_block(stream=None):
    """Context manager yielding (start, end) TimerEvents; end recorded on exit."""

    class _Block:
        def __init__(self):
            self.start = TimerEvent()
            self.end = TimerEvent()

        def __enter__(self):
            self.start.record(stream)
            return self

        def __exit__(self, exc_type, exc, tb):
            self.end.record(stream)
            return False

        @property
        def ms(self):
            """Elapsed milliseconds for the block."""
            return self.start.elapsed_ms(self.end)

    return _Block()
