"""Boot the voice API:  python serve.py [--host 0.0.0.0] [--port 8003]

(Named `serve.py`, not `server.py`: a root-level `server.py` would shadow
the `server/` package on sys.path and break every `server.*` import.)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description="voice-pipeline API server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8003)
    args = ap.parse_args()

    import uvicorn

    from server.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
