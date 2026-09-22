"""Download all voice-pipeline weights.

- whisper-base + Qwen3-0.6B + Kokoro-82M + MiniCPM5-1B (tokenizer) via
  huggingface_hub.snapshot_download (HF cache)
- MiniCPM5-1B Q4_K_M GGUF (default LLM backend) via snapshot_download
  with allow-list (657MB, not the full repo)
- Silero VAD ONNX via stdlib urllib to src/models/engines/silero_vad/silero_vad.onnx
   (matches the ``VadConfig.onnx_path`` default)

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
    "openbmb/MiniCPM5-1B",  # tokenizer + config (weights come from GGUF below)
]

HF_PATTERNS = {
    # repo -> allow_patterns (keep the default-LLM fetch small)
    "openbmb/MiniCPM5-1B-GGUF": ["*Q4_K_M*"],
}

SILERO_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/"
    "silero_vad.onnx"
)
SILERO_REL = os.path.join("src", "models", "engines", "silero_vad", "silero_vad.onnx")


def download_hf(repo, patterns=None):
    from huggingface_hub import snapshot_download

    if patterns:
        path = snapshot_download(repo, allow_patterns=patterns)
    else:
        path = snapshot_download(repo)
    print(f"hf {repo} -> {path}")
    return path


def download_silero(root):
    dest = os.path.join(root, SILERO_REL)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"silero onnx exists: {dest} ({os.path.getsize(dest)}B)")
        return dest
    print(f"fetching silero onnx -> {dest}", flush=True)
    urllib.request.urlretrieve(SILERO_URL, dest)
    print(f"silero onnx done: {dest} ({os.path.getsize(dest)}B)")
    return dest


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
    try:
        download_silero(root)
    except Exception as exc:
        print(f"FAILED silero onnx: {exc}")
        failures.append("silero-vad-onnx")
    if failures:
        print(f"OMITTED/FAILED: {failures}")
        return 1
    print("all models downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
