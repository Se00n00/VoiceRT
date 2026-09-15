"""CUDA-event timing with proper synchronization.

Never benchmark async CUDA ops without sync. Uses torch.cuda.Event when
available, otherwise falls back to perf_counter. Warmup is excluded from
measurement, and enough iterations are required for stable stats.
"""
import time
from typing import Callable, List

import torch


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latencies(
    fn: Callable[[], None],
    warmup: int = 20,
    iterations: int = 100,
    use_cuda_events: bool = True,
) -> List[float]:
    """Run `fn()` warmup times (untimed), then measure `iterations` latencies.

    Returns list of seconds, one per iteration. Caller must ensure `fn`
    does not include model loading/compilation.
    """
    # Warmup (exclude from timing, no events)
    for _ in range(max(0, warmup)):
        fn()
    cuda_sync()

    # If no CUDA or caller disables events, use perf_counter with sync
    if not torch.cuda.is_available() or not use_cuda_events:
        out: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn()
            cuda_sync()
            out.append(time.perf_counter() - t0)
        return out

    # CUDA-event path (precise GPU timing)
    latencies: List[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) / 1000.0)
    return latencies


def measure_with_timestamps(
    stream_fn: Callable[[], List[float]],
    warmup: int = 20,
    iterations: int = 30,
) -> List[List[float]]:
    """For LLM decode: `stream_fn` returns per-token timestamps (seconds).

    Warmup runs are discarded. Returns list of timestamp lists.
    """
    for _ in range(max(0, warmup)):
        stream_fn()
    cuda_sync()
    out: List[List[float]] = []
    for _ in range(iterations):
        ts = stream_fn()
        cuda_sync()
        out.append(ts)
    return out
