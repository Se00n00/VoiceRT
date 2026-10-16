#!/usr/bin/env python3
"""Graphs for worker + long-context benches from measured data (no GPU needed).

Reads benchmarks/results/worker_profile_*.json and longctx_*.json and renders:
  1. worker_sweep.png — TTFT + wall bars per (prompt, gen) cell, decode tok/s line
  2. longctx_scale.png — wall vs prompt ids (prefill scaling)
  3. longctx_needle.png — recall rate per length x depth (grouped bars)

Single smoke points render as single bars; full sweeps extend the same
charts. Re-run after every bench run.

Usage: .venv/bin/python benchmarks/plot_worker_longctx.py
Output: benchmarks/results/plots/*.png @300dpi.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results", "plots")

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def _load(pattern):
    """All rows from every envelope matching pattern, oldest first.

    Missing files read as empty; envelopes always win over crashes.
    ..
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "results", pattern))):
        try:
            with open(path, encoding="utf-8") as fh:
                rows.extend(json.load(fh).get("rows", []))
        except Exception:
            pass
    return rows


def _med(vals):
    """Median of a list, else 0.0. Never raises on empty.

    ..
    """
    s = sorted(vals)
    return float(s[len(s) // 2]) if s else 0.0


def worker_sweep(rows, path):
    """TTFT + wall bars per cell with decode tok/s on a twin axis.

    One group per (prompt, gen); medians over measured reps only.
    ..
    """
    cells = {}
    for r in rows:
        if r.get("rep", 1) == 0:
            continue
        key = (r.get("prompt_target"), r.get("max_tokens"))
        cells.setdefault(key, []).append(r)
    if not cells:
        return False
    keys = sorted(cells, key=lambda k: (k[0] or 0, k[1] or 0))
    labels = [f"p~{k[0]}/g{k[1]}" for k in keys]
    ttft = [_med([c.get("ttft_s", 0.0) for c in cells[k]]) for k in keys]
    wall = [_med([c.get("wall_s", 0.0) for c in cells[k]]) for k in keys]
    tps = [_med([c.get("tps", 0.0) for c in cells[k]]) for k in keys]
    x = range(len(keys))
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.6), 3.6))
    w = 0.35
    ax.bar([i - w / 2 for i in x], ttft, w, label="TTFT (s)", color="#4C78A8")
    ax.bar([i + w / 2 for i in x], wall, w, label="wall (s)", color="#F58518")
    for i, v in enumerate(ttft):
        ax.text(i - w / 2, v * 1.03, f"{v:.1f}s", ha="center", fontsize=8)
    ax.set_xticks(list(x), labels, rotation=20, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title("Bonsai worker: TTFT + wall per cell")
    ax2 = ax.twinx()
    ax2.plot(list(x), tps, "o-", color="#54A24B", label="tok/s")
    ax2.set_ylim(0, max(tps + [1.0]) * 1.5)
    for i, v in enumerate(tps):
        ax2.text(i, v * 1.12, f"{v:.1f}", ha="center", fontsize=8,
                 color="#2c6e2e")
    ax2.set_ylabel("decode tok/s")
    lines, names = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, names + n2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def longctx_scale(rows, path):
    """Wall vs prompt ids, one marker per scale row.

    TTFT overlaid where the leg reports it (fused legs read 0.0).
    ..
    """
    pts = sorted((r for r in rows if r.get("part") == "scale"),
                 key=lambda r: r.get("prompt_ids", 0))
    if not pts:
        return False
    ids = [r.get("prompt_ids", 0) for r in pts]
    wall = [r.get("wall_s", 0.0) for r in pts]
    ttft = [r.get("ttft_s", 0.0) for r in pts]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ids, wall, "o-", label="wall (s)", color="#F58518")
    if any(v > 0 for v in ttft):
        ax.plot(ids, ttft, "s--", label="TTFT (s)", color="#4C78A8")
    for i, v in zip(ids, wall):
        ax.text(i, v * 1.05, f"{v:.1f}s", ha="center", fontsize=8)
    ax.set_ylim(0, max(wall + [1.0]) * 1.35)
    if len(ids) == 1:
        ax.set_xlim(ids[0] - 100, ids[0] + 100)
    ax.set_xlabel("prompt ids")
    ax.set_ylabel("seconds")
    ax.set_title("Prefill scaling: wall vs prompt length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def longctx_needle(rows, path):
    """Recall rate per length, grouped by depth fraction.

    Single probes read 0/1; repeated probes would average out.
    ..
    """
    cells = {}
    for r in rows:
        if r.get("part") != "needle":
            continue
        key = (r.get("length_target"), r.get("depth"))
        cells.setdefault(key, []).append(1.0 if r.get("recall") else 0.0)
    if not cells:
        return False
    lengths = sorted({k[0] for k in cells})
    depths = sorted({k[1] for k in cells})
    x = range(len(lengths))
    w = min(0.8 / max(len(depths), 1), 0.35)
    fig, ax = plt.subplots(figsize=(max(6, len(lengths) * 1.4), 3.6))
    for j, d in enumerate(depths):
        off = (j - (len(depths) - 1) / 2) * w
        vals = [_med(cells.get((L, d), [0.0])) for L in lengths]
        bars = ax.bar([i + off for i in x], vals, w, label=f"d={d}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                    "HIT" if v >= 1.0 else ("MISS" if v <= 0.0 else f"{v:.2f}"),
                    ha="center", fontsize=8)
    ax.set_xticks(list(x), [f"~{L}" for L in lengths])
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("prompt length")
    ax.set_ylabel("recall rate")
    ax.set_title("Needle recall per length x depth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> int:
    """Render all three charts from results JSONs on disk.

    Missing inputs skip their chart; at least one chart must render.
    ..
    """
    os.makedirs(OUTDIR, exist_ok=True)
    made = []
    if worker_sweep(_load("worker_profile_*.json"),
                    os.path.join(OUTDIR, "worker_sweep.png")):
        made.append("worker_sweep.png")
    lrows = _load("longctx_*.json")
    if longctx_scale(lrows, os.path.join(OUTDIR, "longctx_scale.png")):
        made.append("longctx_scale.png")
    if longctx_needle(lrows, os.path.join(OUTDIR, "longctx_needle.png")):
        made.append("longctx_needle.png")
    print(f"wrote {len(made)}: {', '.join(made) or 'nothing to plot'}",
          flush=True)
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
