"""Shared benchmark harness: seeding, CLI helpers, warmup/iter handling."""
import argparse
import random
import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16", "int8"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="HF model id override")


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--seq-lengths", type=int, nargs="*", default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)


def ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dtype_from_str(s: str):
    s = s.lower().replace("torch.", "")
    m = {
        "fp32": torch.float32, "float32": torch.float32,
        "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }
    if s not in m:
        raise ValueError(f"unknown dtype {s}")
    return m[s]


def dtype_name(dtype) -> str:
    rev = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16", torch.int8: "int8"}
    return rev.get(dtype, str(dtype))


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: List[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # only keep fieldnames
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)


def fail_if_not_correct(metrics: Dict[str, float], atol: float, rtol: float, name: str = "") -> None:
    from benchmarks.common.metrics import check_tolerance
    ok, msg = check_tolerance(metrics, atol=atol, rtol=rtol)
    if not ok:
        raise AssertionError(f"Correctness failed for {name}: {msg} metrics={metrics}")
