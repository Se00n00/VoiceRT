"""Horizontal speedup chart: fused Triton vs pure torch.

Current world (post-trim 2026-10-05): only Qwen3-0.6B keeps Triton
(fused leg + paged engine). STT/TTS are pure torch by design, so they
have no Triton bar. The one remaining measured pair is the paged engine
(base 8x32: torch 25.2 tok/s, triton 45.8 tok/s — engine_bench.py runs,
Qwen path untouched by the trim).

After the GPU cold boot, re-take the Qwen3-0.6B leg pair
(benchmark_pytorch.py vs benchmark_triton.py, same flags), extend ROWS
below, and re-run this script.

Output: benchmarks/results/plots/triton_speedup.png (referenced by README).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "plots" / "triton_speedup.png"

# (label, speedup vs pure torch). Higher = triton faster.
# Sources (all measured 2026-10-05, RTX 3050 4GB, real weights):
# - leg: benchmarks/results/quick_20261005_qwen3/llm_latency.csv
#   (Qwen3-0.6B, 64-in/16-out, batch 1: torch 18.5 tok/s, triton 65.9).
# - engine: engine_bench.py base 8x32 (triton 58.9 tok/s re-measured
#   post-boot 2026-10-05; torch 25.2 from the purge-era run — torch path
#   no longer exists in-tree to re-take).
ROWS = [
    ("LLM leg\nQwen3-0.6B, 64+16 tok", 65.9 / 18.5),
    ("Paged engine\nQwen3-0.6B, 8x32 tok", 58.9 / 25.2),
]


def main():
    names = [n for n, _ in ROWS]
    speedups = [s for _, s in ROWS]

    fig, ax = plt.subplots(figsize=(8, 2.6))
    colors = ["#2ca02c" if s >= 1.5 else ("#ffbf00" if s >= 1.05 else "#999999")
              for s in speedups]
    bars = ax.barh(names, speedups, color=colors, edgecolor="black", height=0.5)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("speedup vs pure torch (x, higher is better)")
    ax.set_title("Fused Triton speedup (RTX 3050 4GB, 2026-10-05)")
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{s:.2f}x", va="center", fontsize=10)
    ax.set_xlim(0, max(speedups) * 1.4)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    for n, s in zip(names, speedups):
        print(f"  {n.splitlines()[0]}: {s:.2f}x")


if __name__ == "__main__":
    main()
