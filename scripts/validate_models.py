"""Validate the voice-pipeline checkout: files + import smoke checks.

- size/sha256 for the Silero ONNX file (configs/vad.yaml `onnx_path`)
- HF cache presence hint for whisper/qwen/kokoro (informational only)
- import smoke: triton_kernels, models.*, runtime, engine.engine,
  server.app (no weight loading, no cuda required)

Run: PYTHONPATH=voice-pipeline python scripts/validate_models.py
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMPORTS = [
    "triton_kernels",
    "triton_kernels.qwen",
    "triton_kernels.whisper",
    "triton_kernels.tts",
    "models.whisper",
    "models.qwen",
    "models.tts",
    "models.silero_vad",
    "runtime",
    "engine.engine",
    "server.app",
    "server.routes",
    "server.schemas",
]


def sha256(path, limit_mb=64):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ok = True
    onnx = os.path.join(ROOT, "models", "silero_vad", "silero_vad.onnx")
    if os.path.exists(onnx):
        print(f"silero onnx: {onnx} {os.path.getsize(onnx)}B "
              f"sha256={sha256(onnx)[:16]}...")
    else:
        print(f"MISSING silero onnx: {onnx} (run scripts/download_models.py)")
        ok = False

    try:
        from huggingface_hub import scan_cache_dir  # type: ignore

        repos = {c.repo_id for c in scan_cache_dir().repos}
        for want in ("openai/whisper-base", "Qwen/Qwen2.5-0.5B-Instruct",
                     "hexgrad/Kokoro-82M"):
            print(f"hf cache {'HIT ' if want in repos else 'MISS'} {want}")
            if want not in repos:
                ok = False
    except Exception as exc:
        print(f"hf cache check skipped ({exc})")

    for mod in IMPORTS:
        try:
            __import__(mod)
            print(f"import OK   {mod}")
        except Exception as exc:
            print(f"import FAIL {mod}: {exc}")
            ok = False
    print("validate: OK" if ok else "validate: INCOMPLETE (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
