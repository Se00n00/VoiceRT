#!/usr/bin/env python3
"""High-quality voice-pipeline graphs from measured data (no GPU needed).

Reads benchmarks/results/voice_lazy_20261002.json and renders:
  1. turn_anatomy.png — where the seconds go per turn kind (log scale)
  2. toks_comparison.png — decode tok/s across legs (log scale)
  3. footprint.png — VRAM + RAM footprint vs capacity (linear, cap lines)

Usage: .venv/bin/python benchmarks/plot_voice_lazy.py
Output: benchmarks/results/plots_voice/*.png @300dpi + regenerated note.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results", "voice_lazy_20261002.json")
OUTDIR = os.path.join(HERE, "results", "plots_voice")

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def _annotate_barh(ax, vals, labels, fmt="{:.2g}s"):
    for v, lab in zip(vals, labels):
        ax.text(v * 1.06, lab, fmt.format(v), va="center", fontsize=8)


def turn_anatomy(d, path):
    # (label, segments[(name, seconds)])
    chat = [("route+reply", d["chat_turn_front_s"]["total"])]
    fy = d["first_yes_turn_s"]
    first_yes = [("route", fy["route"]),
                 ("spawn+load+prefill", fy["first_output"] - fy["route"]),
                 ("tools", fy["first_action"] - fy["first_output"]),
                 ("tail", fy["done"] - fy["first_action"])]
    steady = [("full worker turn (completed)", d["steady_yes_turn_s"]["done_completed"])]
    fast = [("fast path E2E", d["fast_path_e2e_s"])]
    rows = [("chat (front)", chat), ("first YES (cold worker)", first_yes),
            ("steady YES (warm worker)", steady), ("fast path", fast)]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = plt.cm.tab10.colors
    for i, (label, segs) in enumerate(rows):
        left = 0.0
        for j, (name, sec) in enumerate(segs):
            ax.barh(label, sec, left=left, color=colors[j % len(colors)],
                    edgecolor="black", linewidth=0.4,
                    label=name if i == 1 else None)
            if sec > 0.5:
                ax.text(left + sec / 2, label, f"{name}\n{sec:.3g}s",
                        ha="center", va="center", fontsize=7)
            left += sec
    ax.set_xscale("log")
    ax.set_xlabel("seconds (log scale)", labelpad=8)
    ax.set_title("Turn anatomy — Qwen3-0.6B front, Bonsai worker CPU\n"
                 "RTX 3050, 2026-10-02", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def toks_comparison(d, path):
    t = d["decode_tps"]
    items = [("front fused\nCUDA 0.6B", t["front_fused_cuda_qwen06"]),
             ("Qwen3-1.7B\nCPU Q4_K_M", t["qwen17_cpu_q4k"]),
             ("Bonsai 27B\nCPU ternary", t["bonsai_cpu_ternary"])]
    labels = [a for a, _ in items]
    vals = [b for _, b in items]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar(labels, vals, color=["#2ca02c", "#ff7f0e", "#d62728"],
                  edgecolor="black", linewidth=0.4)
    ax.axhline(t["historical_batched_claim"], color="gray", linestyle="--",
               linewidth=1, label="historical batched claim (69 tok/s)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:g} tok/s",
                ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("decode tok/s (log scale)")
    ax.set_title("Decode throughput by leg — greedy, batch-1 (2026-10-02)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def footprint(d, path):
    v, r = d["vram_mb"], d["ram_gb"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    cats = ["boot\nlazy", "pre-YES", "post worker\nturn"]
    vals = [v["boot_lazy"], v["pre_yes"], v["post_worker_turn"]]
    bars = a1.bar(cats, vals, color=["#1f77b4", "#ff7f0e", "#d62728"],
                  edgecolor="black", linewidth=0.4)
    a1.axhline(v["card"], color="black", linestyle="-", linewidth=1,
               label=f"card {v['card']}MB")
    a1.annotate(f"TTS OOM: {v['tts_oom_free']}MB free",
                xy=(2, vals[2]), xytext=(1.1, vals[2] + 500),
                fontsize=7, color="red",
                arrowprops={"arrowstyle": "->", "color": "red"})
    for b, x in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, x + 60, f"{x}MB",
                ha="center", fontsize=8)
    a1.set_ylabel("VRAM MB")
    a1.set_title("VRAM footprint")
    a1.legend(fontsize=7)
    rcats = ["avail\nboot", "avail\n+worker", "need\n(16K ctx)"]
    rvals = [r["avail_boot"], r["avail_worker_resident"], r["bonsai_need_16k"]]
    bars = a2.bar(rcats, rvals, color=["#1f77b4", "#ff7f0e", "#d62728"],
                  edgecolor="black", linewidth=0.4)
    for b, x in zip(bars, rvals):
        a2.text(b.get_x() + b.get_width() / 2, x + 0.12, f"{x}GB",
                ha="center", fontsize=8)
    a2.set_ylabel("RAM GB")
    a2.set_title("RAM vs worker need")
    fig.suptitle("Footprint: lazy boot vs co-resident worker (2026-10-02)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    turn_anatomy(d, os.path.join(OUTDIR, "turn_anatomy.png"))
    toks_comparison(d, os.path.join(OUTDIR, "toks_comparison.png"))
    footprint(d, os.path.join(OUTDIR, "footprint.png"))
    print(f"wrote 3 plots -> {OUTDIR}/")


if __name__ == "__main__":
    main()
