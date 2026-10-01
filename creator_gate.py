"""creator_gate: Creator Bot pairing poller for guided onboarding.

The user scans ONE QR (t.me/<CreatorBot>?start=<PAIRING_KEY>) and the
whole bot setup happens conversationally in that chat: name -> BotFather
guide -> paste token -> wired. See src/agent/pairing.py for the flow. ::

    CREATOR_BOT_TOKEN=... .venv/bin/python creator_gate.py   # stdlib only

Env: CREATOR_BOT_TOKEN (or token saved via POST /tg/creator-token).
Exits 2 without a token. Never logs tokens.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import pairing  # noqa: E402
from src.agent.tg_poll import api_call, run_poll  # noqa: E402

TOKEN = pairing.creator_token()


def _log(*parts):
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def _on_text(chat_id: str, text: str) -> None:
    reply = pairing.handle_creator_message(chat_id, text)
    if reply:
        api_call(TOKEN, "sendMessage",
                 {"chat_id": chat_id, "text": reply[:4000]})
        _log("replied to", chat_id[:3] + "…")


def main() -> int:
    if not TOKEN:
        _log("missing CREATOR_BOT_TOKEN (create a bot via @BotFather, "
             "save it with POST /tg/creator-token)")
        return 2
    _log("creator pairing gate starting")
    return run_poll(TOKEN, float(os.environ.get("TG_POLL_S") or 25),
                    _on_text, _log)


if __name__ == "__main__":
    sys.exit(main())
