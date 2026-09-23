#!/usr/bin/env python3
"""Fetch the VoiceAgent and say 'hii'.

Application-layer demo (not core): the agent keeps sessions in memory
only, so THIS file owns persistence — one JSON dotfile per session at
``.voicert/<session_id>.json``. Any app with conversations (TUI, server,
bridge) must add the same load-on-start / save-after-turn pair, e.g. via
:meth:`VoiceAgent.import_session` / :meth:`VoiceAgent.export_session`.

Usage:
  PYTHONPATH=. .venv/bin/python examples/say_hii.py [--text hii]
"""

import argparse
import asyncio
import json
import os
import re
import sys

STORE_DIR = ".voicert"


def session_file(session_id: str) -> str:
    """Dotfile for one session. Never escapes the store dir."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id))[:64] or "session"
    return os.path.join(STORE_DIR, safe + ".json")


def load_session(agent, session_id: str) -> None:
    """Restore a session's window from its dotfile into the agent."""
    try:
        with open(session_file(session_id), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    agent.import_session(session_id, data)


def save_session(agent, session_id: str) -> None:
    """Write a session's window to its dotfile (atomic). Never raises."""
    try:
        os.makedirs(STORE_DIR, exist_ok=True)
        tmp = session_file(session_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(agent.export_session(session_id), f)
        os.replace(tmp, session_file(session_id))
    except Exception as exc:
        print(f"[warn] session save failed: {exc}", file=sys.stderr)


async def _main_async(text: str, session_id: str) -> int:
    from src.main import VoiceAgent

    ag = VoiceAgent()
    load_session(ag, session_id)
    reply = await ag(text, session_id=session_id)
    save_session(ag, session_id)
    print(f"agent: {reply}")
    if not reply:
        print("agent said nothing.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch VoiceAgent and say hii.")
    ap.add_argument("--text", default="hii")
    ap.add_argument("--session-id", default="hii-demo")
    args = ap.parse_args()
    return asyncio.run(_main_async(args.text, args.session_id))


if __name__ == "__main__":
    raise SystemExit(main())
