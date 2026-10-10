"""GAIA validation adapter: gated HF set -> harness manifest (L1+L2+L3).

Needs access first (one-time, human):
  1. accept terms at https://huggingface.co/datasets/gaia-benchmark/GAIA
  2. export HF_TOKEN=<read token>
Then:
  PYTHONPATH=. .venv/bin/python -u benchmarks/gaia_adapter.py
Writes benchmarks/gaia.jsonl: [{id, level, prompt, setup,
attachments, oracle}] where attachments are repo-relative source paths
under benchmarks/gaia/ (copied per task by make_workdir) and the oracle
is reply_match (normalized exact match, GAIA's official rule).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = "gaia-benchmark/GAIA"
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaia")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "gaia.jsonl")


def fetch() -> str:
    """Snapshot the validation split. Returns the local dir.

    ..
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, repo_type="dataset", local_dir=LOCAL,
                             allow_patterns=["2023/validation/*"])


def normalize_answer(text: str) -> str:
    """GAIA official rule: exact match after strip. Pure.

    ..
    """
    return str(text or "").strip()


def adapt(meta_path: str, split_dir: str) -> list:
    """Parquet rows -> harness tasks (levels 1+2+3, file order kept).

    Schema-tolerant: validation parquet mirrors metadata.jsonl
    (task_id/Question/Level/Final answer/file_name), candidate keys
    cover renames. Attachment missing -> task kept, noted.
    ..
    """
    import pandas as pd

    df = pd.read_parquet(meta_path)
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return df[cols[n]]
        return None
    ids = col("task_id", "id")
    questions = col("question")
    levels = col("level")
    answers = col("final_answer", "final-answer", "answer")
    files = col("file_name", "filename", "file")
    tasks = []
    for i in range(len(df)):
        q = str(questions.iloc[i] or "").strip()
        if not q:
            continue
        try:
            level = int(str(levels.iloc[i] or "1").strip()[-1])
        except Exception:
            level = 1
        att, missing = [], []
        fname = str(files.iloc[i] or "").strip() if files is not None else ""
        if fname and fname.lower() != "none":
            src = os.path.join("benchmarks", "gaia", "2023", "validation",
                               os.path.basename(fname))
            (att if os.path.isfile(
                os.path.join(os.path.dirname(OUT), "gaia", "2023",
                             "validation", os.path.basename(fname)))
             else missing).append(src)
        tasks.append({
            "id": f"gaia-{str(ids.iloc[i] or i)[:8]}",
            "level": level,
            "prompt": q,
            "setup": {},
            "attachments": att,
            "attachment_missing": missing,
            "oracle": [{"type": "reply_match",
                        "text": normalize_answer(
                            answers.iloc[i] if answers is not None else "")}],
        })
    return tasks


def main() -> None:
    split = fetch()
    meta = os.path.join(split, "2023", "validation", "metadata.parquet")
    tasks = adapt(meta, os.path.join(split, "2023", "validation"))
    with open(OUT, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n")
    lv = {}
    for t in tasks:
        lv[t["level"]] = lv.get(t["level"], 0) + 1
    miss = sum(len(t["attachment_missing"]) for t in tasks)
    print(f"wrote {OUT}: {len(tasks)} tasks, levels {lv}, "
          f"{miss} missing attachments")


if __name__ == "__main__":
    main()
