"""Voice-term bridge entrypoint: ``PYTHONPATH=. .venv/bin/python server.py``.

All routes live in ``src/server/routes/`` (system, sessions, voice, turns);
this file only parses flags and serves the app on :8004.
"""

from src.server import app

if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="voice-term bridge for Ink TUI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8004)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
