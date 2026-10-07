"""Backend entrypoint: ``PYTHONPATH=. .venv/bin/python -m src.server`` (:8004)."""

import argparse

import uvicorn

from . import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="voice-term bridge for Ink TUI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8004)
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
