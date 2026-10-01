"""Live test for the human contact channel (Telegram bot by default).

Needs a bot token + one /start (open /tg/connect and follow it)::

    TELEGRAM_BOT_TOKEN=... .venv/bin/python -u examples/notify_test.py

    .venv/bin/python -u examples/notify_test.py --phone +917366973856 \\
        --channels whatsapp,voice,ask --ask-timeout 120

The voice leg synthesises --text via the local voice server /tts/say and
sends it as a voice note. The ask leg waits for your REAL reply
(or --fake-reply "..." to resolve in-process).

``--provider baileys`` keeps the WhatsApp-gateway path (needs wa-gate +
QR link); ``callmebot`` the legacy path (needs CONTACT_CALLMEBOT_API_KEY;
voice unsupported there). The call leg is deferred and reports SKIP.
"""
import argparse
import asyncio
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import contacts  # noqa: E402

SERVER = "http://127.0.0.1:8003"


async def _voice_leg(phone: str, text: str) -> tuple[bool, str]:
    """Server /tts/say -> ogg/opus -> gateway voice note."""
    import httpx

    try:
        r = await asyncio.to_thread(
            httpx.post, SERVER + "/tts/say", json={"text": text[:500]},
            timeout=120.0)
        data = r.json() if r.status_code == 200 else {}
    except Exception as exc:
        return False, f"voice synth: server /tts/say unreachable: {exc}"[:200]
    if data.get("kind") != "audio" or not data.get("wav_b64"):
        return False, f"voice synth failed: {str(data)[:150]}"
    try:
        import numpy as np

        wav = np.frombuffer(base64.b64decode(data["wav_b64"]),
                            dtype=np.float32)
        ogg = contacts.encode_ogg_opus(wav, int(data.get("sr", 24000)))
    except RuntimeError as exc:
        return False, f"ogg encode: {exc}"[:200]
    except Exception as exc:
        return False, f"ogg encode failed: {exc}"[:200]
    ok, reason = await contacts.send_whatsapp_voice(
        phone, ogg_b64=base64.b64encode(ogg).decode())
    return ok, reason


async def _ask(phone: str, question: str, timeout_s: float,
               fake_reply: str | None) -> tuple[bool, str]:
    if fake_reply is None:
        print("  (reply in the bot chat, or curl the ask_id to "
              "POST /contact/reply {ask_id, answer})")
        ask_id, answer, status = await contacts.ask_on_whatsapp(
            phone, question, timeout_s)
        print(f"  ask_id={ask_id}")
        if status == "ok":
            return True, f"answer={answer!r}"[:200]
        return False, status

    # fake-reply self-test: run the ask, answer it in-process.
    seen = set(contacts._pending_asks)
    task = asyncio.ensure_future(
        contacts.ask_on_whatsapp(phone, question, timeout_s))

    async def _resolve():
        for _ in range(300):
            await asyncio.sleep(0.1)
            fresh = [k for k in contacts._pending_asks if k not in seen]
            if fresh:
                contacts.receive_answer(fresh[0], fake_reply)
                print(f"  ask_id={fresh[0]} (fake reply landed)")
                return

    await _resolve()
    ask_id, answer, status = await task
    if status == "ok":
        return True, f"answer={answer!r}"[:200]
    return False, status


async def main() -> int:
    ap = argparse.ArgumentParser(description="Contact-channel live test")
    ap.add_argument("--phone", default="+917366973856")
    ap.add_argument("--provider", default="telegram",
                    help="telegram (default) | baileys | callmebot")
    ap.add_argument("--channels", default="whatsapp,voice,ask",
                    help="comma subset of whatsapp,voice,call,ask")
    ap.add_argument("--text", default="VoiceAgent test: pipeline live.")
    ap.add_argument("--question",
                    default="VoiceAgent test question: reply anything.")
    ap.add_argument("--ask-timeout", type=float, default=120.0)
    ap.add_argument("--fake-reply", default=None,
                    help="resolve the ask wait in-process (no phone needed)")
    args = ap.parse_args()

    os.environ["CONTACT_PROVIDER"] = args.provider
    chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    who = contacts.normalize_phone(args.phone) or args.phone
    print(f"provider={args.provider} phone={who} channels={chans}")
    if args.provider == "baileys":
        st = await contacts.gateway_status()
        print(f"gateway: {st}")
        if not st.get("linked"):
            print("  wa-gate not linked: start it (node wa-gate/src/index.js)"
                  " and scan http://localhost:8003/wa/connect")
    elif args.provider == "telegram":
        st = await contacts.telegram_status()
        print(f"telegram: {st}")
        if not st.get("linked"):
            print("  link it: TELEGRAM_BOT_TOKEN=... + press /start "
                  "(see http://localhost:8003/tg/connect); "
                  "tg_gate.py forwards replies")
    elif not os.environ.get("CONTACT_CALLMEBOT_API_KEY"):
        print("CONTACT_CALLMEBOT_API_KEY is unset — legs will report, "
              "not deliver.")

    labels = {"whatsapp": "text message", "voice": "voice note",
              "call": "phone call (TTS)", "ask": "ask + reply"}
    results: dict[str, tuple[bool, str]] = {}
    for i, ch in enumerate(chans, 1):
        print(f"{i}/{len(chans)} {labels.get(ch, ch)} ...")
        if ch == "whatsapp":
            results[ch] = await contacts.send_whatsapp_text(args.phone,
                                                            args.text)
        elif ch == "voice":
            results[ch] = await _voice_leg(args.phone, args.text)
        elif ch == "call":
            results[ch] = await contacts.call_and_speak(args.phone, args.text)
        elif ch == "ask":
            results[ch] = await _ask(args.phone, args.question,
                                     args.ask_timeout, args.fake_reply)
        else:
            results[ch] = (False, f"unknown channel: {ch}")
        ok, reason = results[ch]
        print(f"  -> {'OK' if ok else 'FAIL'}: {reason}")

    print("\nskipped-expected: call (deferred), "
          "voice on provider=callmebot (unsupported)")
    print("summary:")
    code = 0
    for ch, (ok, reason) in results.items():
        tag = "OK" if ok else "FAIL"
        if ch == "call" and "deferred" in reason:
            tag = "SKIP"
        elif (ch == "voice" and args.provider == "callmebot"
                and "unsupported" in reason):
            tag = "SKIP"
        elif not ok:
            code = 1
        print(f"  {ch}: {tag}: {reason}")
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
