"""Statistical + correctness metrics."""
import math
from typing import List, Dict, Tuple

import numpy as np
import torch


def latency_stats(latencies: List[float]) -> Dict[str, float]:
    """latencies in seconds -> median/mean/p50/p95/p99/min/max."""
    if not latencies:
        return {k: 0.0 for k in ("median_ms","mean_ms","p50_ms","p95_ms","p99_ms","min_ms","max_ms","count")}
    arr = np.asarray(latencies, dtype=np.float64) * 1000.0  # ms
    return {
        "count": float(len(arr)),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def throughput_from_latency(latencies: List[float], batch: int = 1, tokens: int = 1) -> Dict[str, float]:
    """Throughput in items/sec from latencies."""
    if not latencies:
        return {"throughput": 0.0}
    mean_s = float(np.mean(latencies))
    if mean_s <= 0:
        return {"throughput": 0.0}
    # items = batch * tokens per iteration
    items = batch * tokens
    return {"throughput": items / mean_s}


def correctness_metrics(a, b, atol: float = 1e-5, rtol: float = 1e-3) -> Dict[str, float]:
    """Compare two tensors/arrays: max_abs, mean_abs, relative, cosine."""
    # Convert to numpy float64 for stable math
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().float().cpu().numpy()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        # try to broadcast - else mark mismatch
        try:
            b = np.broadcast_to(b, a.shape)
        except Exception:
            return {"max_abs_error": float("inf"), "mean_abs_error": float("inf"), "rel_error": float("inf"), "cosine_similarity": 0.0, "shape_mismatch": 1.0}
    diff = np.abs(a - b)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    # relative: mean |diff| / (mean |b| + eps)
    denom = float(np.mean(np.abs(b)) + 1e-9)
    rel = mean_abs / denom if denom else 0.0
    # cosine similarity (flatten)
    af = a.ravel()
    bf = b.ravel()
    denom_cos = (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)
    cos = float(np.dot(af, bf) / denom_cos) if af.size else 1.0
    # NaN/Inf check
    has_nan = bool(np.isnan(a).any() or np.isnan(b).any())
    has_inf = bool(np.isinf(a).any() or np.isinf(b).any())
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_error": float(rel),
        "cosine_similarity": float(cos),
        "has_nan": float(has_nan),
        "has_inf": float(has_inf),
        "within_atol": float(max_abs <= atol),
        "within_rtol": float(rel <= rtol),
    }


def check_tolerance(metrics: Dict[str, float], atol: float = 1e-3, rtol: float = 1e-3) -> Tuple[bool, str]:
    if metrics.get("has_nan"):
        return False, "NaN detected"
    if metrics.get("has_inf"):
        return False, "Inf detected"
    if metrics["max_abs_error"] > atol and metrics["relative_error"] > rtol:
        return False, f"max_abs {metrics['max_abs_error']:.2e} > {atol:.2e} and rel {metrics['relative_error']:.2e} > {rtol:.2e}"
    return True, "ok"
