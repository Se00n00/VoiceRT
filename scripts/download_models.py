"""Download all voice-pipeline weights.

- whisper-base + Qwen2.5-0.5B-Instruct + Kokoro-82M via
  huggingface_hub.snapshot_download (HF cache)
- Silero VAD ONNX via stdlib urllib to models/silero_vad/silero_vad.onnx
  (matches configs/vad.yaml `onnx_path`)

Run: PYTHONPATH=voice-pipeline python scripts/download_models.py [--dest DIR]
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_REPOS = [
    "openai/whisper-base",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "hexgrad/Kokoro-82M",
]

SILERO_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/"
    "silero_vad.onnx"
)
SILERO_REL = os.path.join("models", "silero_vad", "silero_vad.onnx")


def download_hf(repo):
    from huggingface_hub import snapshot_download

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
    root = args.dest or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    failures = []
    for repo in HF_REPOS:
        try:
            download_hf(repo)
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
