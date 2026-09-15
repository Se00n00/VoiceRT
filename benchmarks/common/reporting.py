"""Reporting helpers: summary tables, CSV/JSON aggregation."""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def speedup_pytorch_triton(pytorch_lat_ms: float, triton_lat_ms: float) -> float:
    if triton_lat_ms <= 0:
        return 0.0
    return pytorch_lat_ms / triton_lat_ms  # >1 means Triton faster for latency


def speedup_throughput(triton_tps: float, pytorch_tps: float) -> float:
    if pytorch_tps <= 0:
        return 0.0
    return triton_tps / pytorch_tps


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def aggregate_summary(results_dir: Path) -> Dict[str, Any]:
    """Collectors for summary.json - looks for known CSVs."""
    out: Dict[str, Any] = {}
    for name in ["llm_latency.csv","llm_throughput.csv","llm_memory.csv",
                 "stt_latency.csv","stt_throughput.csv","stt_memory.csv",
                 "tts_latency.csv","tts_throughput.csv","tts_memory.csv"]:
        p = results_dir / name
        if p.exists():
            out[name] = load_csv(p)[:5]  # sample
    return out


def plot_two_series(
    x, y_torch, y_triton, xlabel, ylabel, title, out_path: Path, label_torch="PyTorch", label_triton="Triton"
):
    plt.figure(figsize=(8,5))
    plt.plot(x, y_torch, marker="o", label=label_torch)
    plt.plot(x, y_triton, marker="s", label=label_triton)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_speedup(x, speedups, xlabel, title, out_path: Path):
    plt.figure(figsize=(8,5))
    plt.bar([str(v) for v in x], speedups)
    plt.axhline(1.0, color="red", linestyle="--", label="parity")
    plt.xlabel(xlabel)
    plt.ylabel("Speedup (PyTorch / Triton) >1 = Triton faster")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_summary_table(rows: List[Dict[str, Any]], path: Path):
    """rows: Model | Metric | PyTorch | Triton | Speedup"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # also json
    import json
    with open(path.with_suffix(".json"), "w") as jf:
        json.dump(rows, jf, indent=2)
