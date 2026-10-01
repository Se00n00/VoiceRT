"""tg_gate: Telegram reply poller for the VoiceAgent contact channel.

Long-polls Bot API getUpdates and forwards operator messages to the
Python server so ask_on_whatsapp waits resolve::

    TELEGRAM_BOT_TOKEN=... .venv/bin/python tg_gate.py   # stdlib only

Env: TELEGRAM_BOT_TOKEN (required), CONTACT_INBOUND_URL
(default http://127.0.0.1:8003/contact/inbound), TG_POLL_S (25).

/start from the operator gets a hello reply; every other private text is
forwarded as {phone: "tg:<chat_id>", text}. The server persists the chat
id (see contact_inbound). Never crashes — backoffs on errors, exits only
on missing token (code 2) or KeyboardInterrupt.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.tg_poll import api_call, run_poll  # noqa: E402

TOKEN = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
if not TOKEN:
    # token saved via POST /tg/token lives here (gitignored)
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sessions", "tg", "bot.json")) as _fh:
            TOKEN = str(json.load(_fh).get("token", "") or "").strip()
    except Exception:
        TOKEN = ""
INBOUND = (os.environ.get("CONTACT_INBOUND_URL")
           or "http://127.0.0.1:8003/contact/inbound")

HELLO = ("VoiceAgent live ✅\n"
         "This chat is now linked — questions I send here can be answered "
         "right here.")


def _log(*parts):
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def _post_inbound(phone, text):
    data = json.dumps({"phone": phone, "text": text}).encode()
    req = urllib.request.Request(
        INBOUND, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception as exc:
        _log("inbound forward failed:", str(exc)[:150])
        return False


def _on_text(chat_id: str, text: str) -> None:
    if text.strip() == "/start":
        api_call(TOKEN, "sendMessage", {"chat_id": chat_id, "text": HELLO})
        _log("hello sent to", chat_id)
    _post_inbound(f"tg:{chat_id}", text)


def main():
    if not TOKEN:
        _log("missing TELEGRAM_BOT_TOKEN")
        return 2
    return run_poll(TOKEN, float(os.environ.get("TG_POLL_S") or 25),
                    _on_text, _log)


if __name__ == "__main__":
    sys.exit(main())
