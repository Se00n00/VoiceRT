"""GAIA full loader from HF (parquet-backed, attachments included)."""
import os

REPO = "gaia-benchmark/GAIA"


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ABORT: GAIA loader needs 'datasets' + 'pyarrow' + 'pandas'. "
            "On Kaggle: pip install datasets pyarrow pandas."
        ) from exc


def load_tasks(levels, cache_dir, split="validation"):
    """Load full GAIA tasks. levels: subset of {1,2,3}. Returns (tasks, meta)."""
    _require_datasets()
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    data_dir = snapshot_download(repo_id=REPO, repo_type="dataset",
                                 cache_dir=cache_dir)
    levels = sorted(set(levels or [1, 2, 3]))
    tasks = []
    for lv in levels:
        ds = load_dataset(data_dir, f"2023_level{lv}", split=split)
        for row in ds:
            tasks.append({
                "id": str(row.get("task_id") or row.get("id") or f"L{lv}-{len(tasks)}"),
                "level": lv,
                "prompt": str(row.get("Question") or ""),
                "answer": str(row.get("Final answer") or ""),
                "file_name": row.get("file_name") or "",
                "file_path": os.path.join(data_dir, row.get("file_path") or ""),
            })
    return tasks, {"repo": REPO, "split": split, "levels": levels,
                   "n": len(tasks), "data_dir": data_dir}
