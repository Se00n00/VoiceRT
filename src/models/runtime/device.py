"""Device selection, sync, and VRAM counters with CPU fallback."""
import contextlib
import threading

import torch


def is_cuda_available():
    """True when a CUDA device can be used."""
    return torch.cuda.is_available()


def select_device(prefer="cuda", index=0):
    """Pick torch.device(prefer:index) when usable, else torch.device('cpu').

    Never raises for a missing GPU: falls back to CPU so library imports
    and CPU-only paths keep working.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
            if count > 0:
                return torch.device(f"cuda:{min(index, count - 1)}")
        except Exception:
            pass
    return torch.device("cpu")


def current_device(index_if_cuda=0):
    """Best default device for this host (cuda:index or cpu)."""
    return select_device("cuda", index_if_cuda)


def synchronize(device=None):
    """Block until device work completes (no-op on CPU-only hosts)."""
    if not torch.cuda.is_available():
        return
    try:
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(torch.device(device))
    except Exception:
        # Synchronize must never break a pipeline stage; a missed fence
        # only affects timing precision, not correctness of results.
        pass


def _bytes_to_mb(nbytes):
    return float(nbytes) / (1024 ** 2)


def allocated_mb(device=None):
    """Currently allocated CUDA memory in MB (0.0 on CPU-only hosts)."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        idx = torch.cuda.current_device() if device is None else torch.device(device).index or 0
        return _bytes_to_mb(torch.cuda.memory_allocated(idx))
    except Exception:
        return 0.0


def reserved_mb(device=None):
    """CUDA memory reserved by the caching allocator in MB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        idx = torch.cuda.current_device() if device is None else torch.device(device).index or 0
        return _bytes_to_mb(torch.cuda.memory_reserved(idx))
    except Exception:
        return 0.0


def max_allocated_mb(device=None):
    """Peak allocated CUDA memory since last reset, in MB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        idx = torch.cuda.current_device() if device is None else torch.device(device).index or 0
        return _bytes_to_mb(torch.cuda.max_memory_allocated(idx))
    except Exception:
        return 0.0


def reset_peak_stats(device=None):
    """Reset CUDA peak counters (no-op on CPU-only hosts)."""
    if not torch.cuda.is_available():
        return
    try:
        if device is None:
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats(torch.device(device).index)
    except Exception:
        pass


def vram_stats(device=None):
    """Dict of allocated/reserved/peak MB plus device name."""
    stats = {
        "allocated_mb": allocated_mb(device),
        "reserved_mb": reserved_mb(device),
        "peak_mb": max_allocated_mb(device),
        "cuda": bool(torch.cuda.is_available()),
        "device": str(select_device("cuda") if device is None else torch.device(device)),
    }
    if torch.cuda.is_available():
        try:
            stats["name"] = torch.cuda.get_device_name(device)
        except Exception:
            stats["name"] = "unknown"
    else:
        stats["name"] = "cpu"
    return stats


def keepalive_start(interval_s: float = 0.002):
    """Hold GPU clocks up across one bursty turn; returns stop().

    Laptop GPUs collapse to idle clocks (~210 MHz here) within tens of ms
    of no work, and every leg of a voice turn (STT burst, LLM prefill,
    TTS burst) would re-pay the ramp. A daemon thread running one tiny
    kernel per ``interval_s`` keeps the clocks hot for the turn's
    duration only — no root, no persistent power burn. stop() is
    idempotent and never raises; on CPU-only hosts both are no-ops.
    ``VOICE_KEEP_MS`` env overrides the interval (0 disables).
    """
    import os as _os

    try:
        interval_s = float(_os.environ.get("VOICE_KEEP_MS", "2")) / 1000.0
    except Exception:
        pass
    if not torch.cuda.is_available() or interval_s <= 0:
        return lambda: None
    stop = threading.Event()

    def _spin():
        try:
            a = torch.zeros(64, 64, device="cuda")
            while not stop.wait(interval_s):
                a.add_(1)
        except Exception:
            pass

    t = threading.Thread(target=_spin, daemon=True)
    t.start()

    def _stop():
        try:
            stop.set()
        except Exception:
            pass

    return _stop


def gpu_rev(iters: int = 4, size: int = 512) -> float:
    """Synchronous clock rev: a short matmul burst forcing P-state up.

    Call once right before a latency-critical GPU phase after idle — the
    ~3-5 ms spent here buys back 10-100 ms of ramp penalty inside the
    phase itself. Returns seconds spent. Never raises; no-op without CUDA.
    """
    import time as _t

    t0 = _t.perf_counter()
    try:
        if not torch.cuda.is_available():
            return 0.0
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        for _ in range(max(1, int(iters))):
            a = b @ a
        torch.cuda.synchronize()
    except Exception:
        pass
    return _t.perf_counter() - t0
