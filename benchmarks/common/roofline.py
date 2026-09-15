"""Roofline helper: arithmetic intensity, achieved TFLOPs/BW, plot."""
import json
import csv
from pathlib import Path
from typing import Dict, List

import torch

# Hardware specs for RTX 3050 Laptop (spec) - can be overridden
HW_SPECS = {
    "rtx3050_laptop": {
        "peak_fp16_tflops": 14.0,  # approximate
        "peak_fp32_tflops": 7.0,
        "peak_bw_gbps": 224.0,  # 128-bit * 14 Gbps /8
        "name": "RTX 3050 Laptop",
    },
    "a100": {
        "peak_fp16_tflops": 312.0,
        "peak_fp32_tflops": 19.5,
        "peak_bw_gbps": 2039.0,
        "name": "A100",
    },
}


def estimate_kernel_flops(kernel: str, shape: Dict) -> float:
    """Analytical FLOPs for known kernels. shape dict contains dims."""
    if kernel == "rmsnorm":
        # var + rsqrt + mul: ~5*N per row? N=hidden
        N = shape.get("N", 1024)
        B = shape.get("B", 4)
        return B * N * 5
    if kernel == "rope":
        # 4 mul + 2 add per element? half rotation: 2 mul +1 sub per pair
        H = shape.get("H", 1024)
        B = shape.get("B", 1)
        return B * H * 4
    if kernel == "swiglu":
        N = shape.get("N", 1024)
        B = shape.get("B", 1024)
        return B * N * 6  # exp+div+mul
    if kernel in ("decode_attn", "gqa_decode_attn", "batched_decode_attn"):
        H = shape.get("H", 8)
        N = shape.get("N", 128)
        D = shape.get("D", 64)
        # QK^T: H*N*D*2, softmax: ~5*H*N, AV: H*N*D*2
        return H * (2 * N * D + 5 * N + 2 * N * D)
    if kernel in ("fused_qkv", "fused_qkv_gqa"):
        K = shape.get("K", 1024)
        D = shape.get("D", 512)
        # 3 GEMV: 2*K*D each
        return 3 * 2 * K * D
    if kernel == "conv1d_silu":
        N, Cin, Cout, L, K = shape.get("N",1), shape.get("Cin",32), shape.get("Cout",32), shape.get("L",200), shape.get("K",3)
        # conv flops: N*Cout*L*Cin*K*2
        return N * Cout * L * Cin * K * 2 + N*Cout*L*3  # silu extra
    if kernel == "in1d_silu":
        N, C, L = shape.get("N",2), shape.get("C",32), shape.get("L",200)
        R = N*C
        return R * (5*L + 3*L)  # mean/var + silu
    if kernel == "layernorm":
        N = shape.get("N", 512)
        B = shape.get("B", 32)
        return B * N * 7
    if kernel == "row_softmax":
        N = shape.get("N", 1500)
        B = shape.get("B", 8)
        return B * N * 5
    return 0.0


def estimate_bytes(kernel: str, shape: Dict, dtype_bytes: int = 2) -> float:
    """Bytes read+written."""
    if kernel == "rmsnorm":
        N, B = shape.get("N",1024), shape.get("B",4)
        # read x + w, write y, plus var intermediate (not mem)
        return B * (N*dtype_bytes*2 + N*dtype_bytes)  # x,w read y write
    if kernel == "rope":
        dh = shape.get("dh",128)
        B = shape.get("B", 4)
        half = dh//2
        return B * (dh*dtype_bytes*1 + half*dtype_bytes*2 + dh*dtype_bytes)  # x + cos/sin + y
    if kernel == "swiglu":
        N, B = shape.get("N",1024), shape.get("B",1024)
        return B * (2*N*dtype_bytes + N*dtype_bytes)  # gate,up read y write
    if kernel in ("decode_attn",):
        H, N, D = shape.get("H",8), shape.get("N",128), shape.get("D",64)
        return H*D*dtype_bytes + 2*H*N*D*dtype_bytes + H*D*dtype_bytes  # q + K,V + out
    if kernel == "layernorm":
        N, B = shape.get("N",512), shape.get("B",32)
        return B * (N*dtype_bytes + 2*N*dtype_bytes + N*dtype_bytes)  # x,w,b,y
    # generic fallback: 2* flops bytes?
    flops = estimate_kernel_flops(kernel, shape)
    return flops * dtype_bytes  # rough


def roofline_record(kernel: str, shape: Dict, median_ms: float, dtype_bytes: int = 2, hw: str = "rtx3050_laptop") -> Dict:
    flops = estimate_kernel_flops(kernel, shape)
    bytes_ = estimate_bytes(kernel, shape, dtype_bytes)
    ai = flops / bytes_ if bytes_ else 0
    # achieved
    t_s = median_ms / 1000.0 if median_ms else 1e-9
    achieved_tflops = (flops / t_s) / 1e12 if t_s else 0
    achieved_bw_gbps = (bytes_ / t_s) / 1e9 if t_s else 0
    spec = HW_SPECS.get(hw, HW_SPECS["rtx3050_laptop"])
    # Determine bound: if ai * peak_bw < peak_flops => mem bound else compute
    peak_flops = spec["peak_fp16_tflops"] if dtype_bytes==2 else spec["peak_fp32_tflops"]
    peak_bw = spec["peak_bw_gbps"]
    roof = min(peak_flops, peak_bw * ai)
    # achieved vs roof -> utilization?
    bound = "memory-bound" if (peak_bw * ai) < peak_flops else "compute-bound"
    # If achieved low, keep label but note
    return {
        "kernel": kernel,
        "flops": flops,
        "bytes": bytes_,
        "arithmetic_intensity": ai,
        "median_ms": median_ms,
        "achieved_tflops": achieved_tflops,
        "achieved_bw_gbps": achieved_bw_gbps,
        "peak_tflops": peak_flops,
        "peak_bw_gbps": peak_bw,
        "roof_tflops": roof,
        "bound": bound,
        "hw": hw,
    }


def write_roofline_csv(records: List[Dict], path: Path):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


def plot_roofline(records: List[Dict], out_path: Path, hw: str = "rtx3050_laptop"):
    import matplotlib.pyplot as plt
    spec = HW_SPECS.get(hw, HW_SPECS["rtx3050_laptop"])
    # Roofline shape: y = min(peak_flops, x*peak_bw)
    xs = [r["arithmetic_intensity"] for r in records]
    ys = [r["achieved_tflops"] for r in records]
    labels = [r["kernel"] for r in records]
    # theoretical roof
    import numpy as np
    x_theory = np.logspace(-1, 2, 100)
    y_theory = [min(spec["peak_fp16_tflops"], spec["peak_bw_gbps"]/1000 * x) for x in x_theory]  # bw in TF? convert
    # Actually bw Gbps -> TB/s: 224 Gbps = 224e9 bytes, flops 14e12, ai in flops/byte, roof = min(peak_flops, bw*ai /1e12)
    # Let's compute correctly: peak_bw 224 GB/s = 224e9 bytes, ai flops/byte => bw*ai flops/s
    y_theory = [min(spec["peak_fp16_tflops"], spec["peak_bw_gbps"] * x / 1000) for x in x_theory]  # rough
    plt.figure(figsize=(8,6))
    plt.loglog(x_theory, y_theory, "k--", label=f"Roof ({spec['name']})")
    for x,y,lab in zip(xs, ys, labels):
        plt.loglog(x, y, "o", label=lab)
        plt.text(x, y, f" {lab}", fontsize=8)
    plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.ylabel("Performance (TFLOP/s)")
    plt.title(f"Roofline ({spec['name']})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
