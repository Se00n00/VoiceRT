"""Memory reporting and budget guards (CPU-safe: report 0.0 without CUDA)."""
import torch

from runtime.device import allocated_mb, max_allocated_mb, reserved_mb


class MemoryBudgetExceeded(MemoryError, RuntimeError):
    """Raised when a planned allocation would exceed the VRAM budget.

    Dual-inherits RuntimeError so server error-mapping (which maps
    RuntimeError leg failures to HTTP 503) catches it without special
    cases; isinstance checks against either parent both succeed.
    """


def _peak_reserved_mb():
    """Peak reserved CUDA memory since last reset, in MB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_reserved()) / (1024 ** 2)
    except Exception:
        return 0.0


def summary():
    """Snapshot dict of allocated/reserved/peak MB."""
    return {
        "allocated_mb": allocated_mb(),
        "reserved_mb": reserved_mb(),
        "peak_mb": max_allocated_mb(),
        "peak_reserved_mb": _peak_reserved_mb(),
        "cuda": bool(torch.cuda.is_available()),
    }


def fits(estimate_mb, budget_mb, headroom_mb=0.0):
    """True if current usage + estimate stays within budget."""
    return (allocated_mb() + float(estimate_mb) + float(headroom_mb)) <= float(budget_mb)


def check_budget(estimate_mb, budget_mb, headroom_mb=0.0, what="allocation"):
    """Raise MemoryBudgetExceeded if estimate does not fit; else return usage dict."""
    usage = summary()
    if not fits(estimate_mb, budget_mb, headroom_mb):
        raise MemoryBudgetExceeded(
            f"{what} needs ~{estimate_mb:.0f}MB but budget is {budget_mb:.0f}MB "
            f"(allocated={usage['allocated_mb']:.0f}MB, "
            f"headroom={headroom_mb:.0f}MB)"
        )
    return usage
