"""Device selection, sync, and VRAM counters with CPU fallback."""
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
