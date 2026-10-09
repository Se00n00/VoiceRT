"""Download all voice-pipeline weights.

- whisper-base + Qwen3-0.6B + Kokoro-82M via
  huggingface_hub.snapshot_download (HF cache)
- Ternary Bonsai 2 27B GGUF (worker LLM backend) via snapshot_download
  with allow-list (PTQ pack, not the full repo)
- Silero VAD ships inside the ``silero-vad`` pip wheel
  (see requirements.txt) — nothing to fetch here.

Run: PYTHONPATH=. python -m src.models.download [--dest DIR]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

HF_REPOS = [
    "openai/whisper-base",
    "Qwen/Qwen3-0.6B",
    "hexgrad/Kokoro-82M",
]

HF_PATTERNS = {
    # Ternary Bonsai 2 27B (worker leg; filenames verified at M0
    # fetch — HF repo: prism-ml/Ternary-Bonsai-2-27B-gguf)
    "prism-ml/Ternary-Bonsai-2-27B-gguf": [
        "*PTQ1_0*",  # 5.93GB small pack (4GB-box default)
        "*PQ2_0*",  # 7.25GB fast-prefill pack (T4+ default)
        "*mmproj*",  # vision tower (HQQ 4-bit)
    ],
}

def download_hf(repo, patterns=None):
    from huggingface_hub import snapshot_download

    if patterns:
        path = snapshot_download(repo, allow_patterns=patterns)
    else:
        path = snapshot_download(repo)
    print(f"hf {repo} -> {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=None,
                    help="repo root override (default: voice-pipeline/)")
    args = ap.parse_args(argv)
    root = args.dest or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    failures = []
    for repo in HF_REPOS:
        try:
            download_hf(repo)
        except Exception as exc:
            print(f"FAILED hf {repo}: {exc}")
            failures.append(repo)
    for repo, patterns in HF_PATTERNS.items():
        try:
            download_hf(repo, patterns)
        except Exception as exc:
            print(f"FAILED hf {repo}: {exc}")
            failures.append(repo)
    if failures:
        print(f"OMITTED/FAILED: {failures}")
        return 1
    print("all models downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
