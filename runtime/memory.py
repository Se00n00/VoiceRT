"""Memory reporting and budget guards (CPU-safe: report 0.0 without CUDA)."""
import torch

from runtime.device import allocated_mb, max_allocated_mb, reserved_mb, reset_peak_stats


class MemoryBudgetExceeded(MemoryError):
    """Raised when a planned allocation would exceed the VRAM budget."""


def allocated():
    """Currently allocated CUDA memory in MB."""
    return allocated_mb()


def reserved():
    """CUDA memory held by the caching allocator in MB."""
    return reserved_mb()


def peak():
    """Peak allocated CUDA memory since last reset, in MB."""
    return max_allocated_mb()


def peak_reserved_mb():
    """Peak reserved CUDA memory since last reset, in MB (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_reserved()) / (1024 ** 2)
    except Exception:
        return 0.0


def reset_peaks():
    """Reset CUDA peak counters."""
    reset_peak_stats()


def summary():
    """Snapshot dict of allocated/reserved/peak MB."""
    return {
        "allocated_mb": allocated(),
        "reserved_mb": reserved(),
        "peak_mb": peak(),
        "peak_reserved_mb": peak_reserved_mb(),
        "cuda": bool(torch.cuda.is_available()),
    }


def fits(estimate_mb, budget_mb, headroom_mb=0.0):
    """True if current usage + estimate stays within budget."""
    return (allocated() + float(estimate_mb) + float(headroom_mb)) <= float(budget_mb)


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


class MemoryBudget:
    """Named VRAM budget guard for one pipeline stage or leg."""

    def __init__(self, budget_mb, headroom_mb=0.0, name="stage"):
        self.budget_mb = float(budget_mb)
        self.headroom_mb = float(headroom_mb)
        self.name = name

    def check(self, estimate_mb):
        """Raise MemoryBudgetExceeded when estimate_mb does not fit."""
        return check_budget(estimate_mb, self.budget_mb, self.headroom_mb, self.name)

    def guard(self, estimate_mb):
        """Context manager that checks the budget on entry."""
        budget = self

        class _Guard:
            def __enter__(self):
                budget.check(estimate_mb)
                return budget

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Guard()

    def usage(self):
        """Current memory snapshot dict."""
        return summary()
