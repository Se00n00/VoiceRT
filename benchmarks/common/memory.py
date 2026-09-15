"""GPU memory helpers — allocated vs reserved, peak vs baseline."""
import torch
from typing import Dict


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def snapshot() -> Dict[str, float]:
    """Current memory in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0, "peak_reserved_mb": 0.0}
    try:
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }
    except Exception:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0, "peak_reserved_mb": 0.0}


def baseline_mb() -> float:
    return snapshot()["allocated_mb"]


def peak_allocated_mb() -> float:
    return snapshot()["peak_mb"]


def format_mb(x: float) -> str:
    return f"{x:.1f} MB"
