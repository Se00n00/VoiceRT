"""Shared helpers for eval/ (no model code, no GPU, stdlib only)."""
import json
import os
import time


def estimate_tokens(text_len_chars: int) -> int:
    return max(1, int(text_len_chars) // 4)


def budget_for(max_seq: int, max_tokens: int, reserve: int = 512) -> int:
    return int(max_seq) - int(max_tokens) - int(reserve)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def write_results(path: str, payload: dict) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_done_keys(path: str) -> set:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", data) if isinstance(data, dict) else data
        out = set()
        for r in rows or []:
            if isinstance(r, dict) and "id" in r:
                out.add(f"{r.get('version', '?')}/{r.get('id', '?')}")
        return out
    except Exception:
        return set()


def hf_download(repo_id: str, filename: str, cache_dir: str, rev: str = "main") -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset",
        revision=rev, cache_dir=cache_dir,
    )


def read_json_lines(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
