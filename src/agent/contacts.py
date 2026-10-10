"""Human contact channel: reach the operator via WhatsApp / phone call.

Default provider is Telegram (``CONTACT_PROVIDER=telegram``): the operator
creates a bot once via @BotFather, presses /start once, and the app owns a
bot chat — text, real voice notes, and real inbound replies, no second SIM.

``CONTACT_PROVIDER=telegram`` (default) | ``baileys`` (self-hosted WhatsApp
gateway in ``wa-gate/``) | ``callmebot`` (legacy text/call fallback behind
``CONTACT_CALLMEBOT_API_KEY``).

Ask/reply loop: :func:`ask_on_whatsapp` sends the question and waits on a
Future keyed by ask_id. Replies land via :func:`receive_inbound` (real
WhatsApp text from the gateway, matched to the newest pending ask from
that phone) or :func:`receive_answer` (explicit ask_id, i.e. the
``POST /contact/reply`` fake-reply path).

Phone calls are DEFERRED: Baileys cannot dial PSTN, so :func:`call_and_speak`
reports that unless ``CONTACT_CALLMEBOT_API_KEY`` is set (CallMeBot TTS
call fallback).

Every public function degrades to ``(False, reason)`` / ``("", None,
reason)`` and never raises — same house rule as the model engines.
"""
import asyncio
import base64
import io
import json
import os
import subprocess
import time
import uuid

import httpx
import numpy as np
import qrcode

__all__ = [
    "normalize_phone",
    "gateway_status",
    "telegram_status",
    "telegram_chat_id",
    "remember_tg_chat",
    "set_tg_token",
    "validate_tg_token_sync",
    "tg_deeplink_qr",
    "register_agent",
    "get_agent_record",
    "send_whatsapp_text",
    "send_whatsapp_voice",
    "encode_ogg_opus",
    "call_and_speak",
    "notify",
    "ask_on_whatsapp",
    "receive_answer",
    "receive_inbound",
]

WA_URL = "https://api.callmebot.com/whatsapp.php"
CALL_URL = "https://api.callmebot.com/call.php"

CALL_DEFERRED = (
    "call leg deferred: the gateway cannot dial PSTN; set "
    "CONTACT_CALLMEBOT_API_KEY to use the CallMeBot TTS-call fallback"
)

# ask_id -> (future, e164 phone asked)
_pending_asks: dict[str, tuple[asyncio.Future, str]] = {}


def provider() -> str:
    return (os.environ.get("CONTACT_PROVIDER") or "telegram").strip().lower()


def gateway_url() -> str:
    return (os.environ.get("CONTACT_GATEWAY_URL")
            or "http://127.0.0.1:8100").rstrip("/")


def normalize_phone(phone: str) -> str:
    """Best-effort E.164: '7366973856' -> '+917366973856'.

    Strips spaces/dashes/dots/parens; keeps a leading '+'; bare 10-digit
    Indian mobiles gain '+91'; '91' + 10 digits gains '+'.
    """
    digits = "".join(c for c in str(phone or "") if c.isdigit())
    if str(phone or "").strip().startswith("+") and digits:
        return "+" + digits
    if len(digits) == 10:
        return "+91" + digits
    if len(digits) == 12 and digits.startswith("91"):
        return "+" + digits
    return "+" + digits if digits else ""


def _api_key() -> str:
    return (os.environ.get("CONTACT_CALLMEBOT_API_KEY") or "").strip()


def _http_get_sync(url: str, params: dict, timeout_s: float) -> tuple[bool, str]:
    """Blocking GET via httpx; monkeypatch seam for unit tests."""

    try:
        r = httpx.get(url, params=params, timeout=timeout_s)
        body = (r.text or "")[:300]
        if r.status_code == 200 and "error" not in body.lower():
            return True, body
        return False, f"http {r.status_code}: {body}"[:300]
    except Exception as exc:
        return False, f"request failed: {exc}"[:300]


def _http_post_sync(url: str, payload: dict,
                    timeout_s: float) -> tuple[bool, str]:
    """Blocking POST-JSON via httpx; monkeypatch seam for unit tests."""

    try:
        r = httpx.post(url, json=payload, timeout=timeout_s)
        try:
            data = r.json()
        except Exception:
            data = {}
        if r.status_code == 200 and isinstance(data, dict) and data.get("ok"):
            return True, str(data.get("err") or "ok")[:300]
        err = (data.get("err") if isinstance(data, dict) else None
               ) or (r.text or "")[:200]
        return False, f"http {r.status_code}: {err}"[:300]
    except Exception as exc:
        return False, f"request failed: {exc}"[:300]


async def _http_get(url: str, params: dict,
                    timeout_s: float = 30.0) -> tuple[bool, str]:
    try:
        return await asyncio.to_thread(_http_get_sync, url, params, timeout_s)
    except Exception as exc:  # executor gone / event loop closing
        return False, f"request failed: {exc}"[:200]


async def _gw_post(path: str, payload: dict,
                   timeout_s: float = 30.0) -> tuple[bool, str]:
    try:
        return await asyncio.to_thread(
            _http_post_sync, gateway_url() + path, payload, timeout_s)
    except Exception as exc:
        return False, f"request failed: {exc}"[:200]


async def gateway_status() -> dict:
    """``wa-gate`` health: {linked, phone, ...} or {linked: False, err}."""
    try:
    
        r = await asyncio.to_thread(
            httpx.get, gateway_url() + "/wa/health", timeout=5.0)
        if r.status_code == 200:
            return dict(r.json())
        return {"linked": False, "err": f"http {r.status_code}"}
    except Exception as exc:
        return {"linked": False, "err": f"gateway down: {exc}"[:200]}


# -- Telegram provider (official Bot API, httpx only) ---------------------

def _tg_token() -> str:
    env = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if env:
        return env
    try:

        with open(_tg_bot_file()) as fh:
            tok = str(json.load(fh).get("token", "") or "").strip()
            if tok:
                return tok
    except Exception:
        pass
    try:
        rec = get_agent_record("default")
        return str((rec or {}).get("bot_token", "") or "").strip()
    except Exception:
        return ""


def _tg_agents_file() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(here, "sessions", "tg", "agents.json")


def register_agent(agent_id: str, name: str, chat_id: str, token: str,
                   username: str) -> bool:
    """Record an agent bot from pairing (tokens stay in sessions/).
    Never raises."""
    try:

        agent_id = str(agent_id or "default")[:64]
        path = _tg_agents_file()
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[agent_id] = {"name": str(name or "")[:64],
                          "chat_id": str(chat_id or ""),
                          "bot_token": str(token or ""),
                          "bot_username": str(username or "")}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh)
        return True
    except Exception:
        return False


def get_agent_record(agent_id: str = "default") -> dict | None:
    """Agent bot record (includes token — never log it). Never raises."""
    try:

        with open(_tg_agents_file()) as fh:
            data = json.load(fh)
            rec = (data or {}).get(str(agent_id or "default"))
            return dict(rec) if isinstance(rec, dict) else None
    except Exception:
        return None


def _tg_bot_file() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(here, "sessions", "tg", "bot.json")


def _tg_bot_info() -> dict:
    try:

        with open(_tg_bot_file()) as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tg_save_bot(token: str, username: str) -> bool:
    try:

        path = _tg_bot_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"token": token, "username": username}, fh)
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        return True
    except Exception:
        return False


async def set_tg_token(token: str) -> tuple[bool, str]:
    """Validate a BotFather token via getMe and persist it (sessions/tg/).
    Returns (True, "@username") or (False, reason). Never raises."""
    token = str(token or "").strip()
    if not token or ":" not in token:
        return False, "that doesn't look like a bot token (want 123:ABC…)"
    try:
        ok, info = await asyncio.to_thread(validate_tg_token_sync, token)
    except Exception as exc:
        return False, f"validation failed: {exc}"[:200]
    if not ok:
        return False, info
    username = info.lstrip("@") if info.startswith("@") else ""
    if not _tg_save_bot(token, username):
        return False, "could not persist token (sessions/ unwritable?)"
    return True, ("@" + username) if username else "ok"


def validate_tg_token_sync(token: str) -> tuple[bool, str]:
    """Blocking getMe validation. Returns (True, "@username") or
    (False, reason). Sync core shared by set_tg_token and pairing
    finalize (which runs outside the event loop). Never raises."""
    try:
        token = str(token or "").strip()
        if not token or ":" not in token:
            return False, "that doesn't look like a bot token (want 123:ABC…)"
        ok, res = _tg_call_sync("getMe", None, None, 15.0, token=token)
        if not ok:
            return False, str(res.get("description", "getMe failed"))[:200]
        username = str(res.get("username", "") or "")
        return True, ("@" + username) if username else "ok"
    except Exception as exc:
        return False, f"validation failed: {exc}"[:200]


def tg_deeplink_qr() -> tuple[bool, str]:
    """PNG data URL of a QR encoding the t.me deep link (scan → bot chat).
    Returns (True, data_url) or (False, reason). Never raises."""
    try:
        username = str(_tg_bot_info().get("username", "") or "").strip()
        if not username and _tg_token():
            # token from env only (never persisted): one live getMe
            ok, res = _tg_call_sync("getMe", None, None, 15.0)
            if ok:
                username = str(res.get("username", "") or "").strip()
        if not username:
            return False, "bot username unknown (set token first)"
        img = qrcode.make(f"https://t.me/{username}")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return True, ("data:image/png;base64,"
                      + base64.b64encode(buf.getvalue()).decode())
    except Exception as exc:
        return False, f"qr failed: {exc}"[:200]


def _tg_state_file() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(here, "sessions", "tg", "chat.json")


def remember_tg_chat(chat_id: str) -> str:
    """Persist the operator chat id (from /start). Never raises."""
    try:
        chat_id = str(chat_id or "").strip()
        if not chat_id:
            return ""
        path = _tg_state_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as fh:
            json.dump({"chat_id": chat_id}, fh)
        return chat_id
    except Exception:
        return ""


def telegram_chat_id() -> str:
    """Operator chat id: env override, else persisted /start chat."""
    env = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if env:
        return env
    try:

        with open(_tg_state_file()) as fh:
            return str(json.load(fh).get("chat_id", "") or "").strip()
    except Exception:
        return ""


def _tg_call_sync(method: str, data: dict | None, files: dict | None,
                  timeout_s: float, token: str | None = None
                  ) -> tuple[bool, dict]:
    """Blocking Bot API call; monkeypatch seam for unit tests.

    Returns (True, result) or (False, {"description": ...}).
    """

    token = (token or _tg_token()).strip()
    if not token:
        return False, {"description": "missing TELEGRAM_BOT_TOKEN"}
    try:
        if files:
            r = httpx.post(f"https://api.telegram.org/bot{token}/{method}",
                           data=data or {}, files=files, timeout=timeout_s)
        else:
            r = httpx.post(f"https://api.telegram.org/bot{token}/{method}",
                           json=data or {}, timeout=timeout_s)
        try:
            body = r.json()
        except Exception:
            body = {}
        if isinstance(body, dict) and body.get("ok"):
            res = body.get("result")
            return True, res if isinstance(res, dict) else {"result": res}
        desc = (body.get("description") if isinstance(body, dict) else None
                ) or (r.text or "")[:200]
        code = r.status_code
        if code == 401:
            desc = "bad TELEGRAM_BOT_TOKEN (401)"
        elif code == 403:
            desc = ("operator must press /start on the bot first "
                    f"(403: {desc})")
        return False, {"description": f"http {code}: {desc}"[:300]}
    except Exception as exc:
        return False, {"description": f"request failed: {exc}"[:200]}


async def _tg_call(method: str, data: dict | None = None,
                   files: dict | None = None,
                   timeout_s: float = 30.0) -> tuple[bool, dict]:
    try:
        return await asyncio.to_thread(_tg_call_sync, method, data, files,
                                       timeout_s)
    except Exception as exc:
        return False, {"description": f"request failed: {exc}"[:200]}


async def telegram_status() -> dict:
    """Bot link state: {linked, chat_id, bot} or {linked: False, err}."""
    if not _tg_token():
        return {"linked": False, "err": "missing TELEGRAM_BOT_TOKEN"}
    ok, res = await _tg_call("getMe", timeout_s=10.0)
    if not ok:
        return {"linked": False,
                "err": str(res.get("description", "getMe failed"))[:200]}
    chat = telegram_chat_id()
    if not chat:
        return {"linked": False, "bot": res.get("username", ""),
                "err": "operator must press /start on the bot first"}
    return {"linked": True, "chat_id": chat,
            "bot": res.get("username", "")}


def _need_key_and_phone(phone: str) -> tuple[str, str, str]:
    """-> (e164, key, err). err empty on success."""
    e164 = normalize_phone(phone)
    if not e164:
        return "", "", "empty phone number"
    key = _api_key()
    if not key:
        return e164, "", ("missing CONTACT_CALLMEBOT_API_KEY "
                          "(opt the number in via WhatsApp first)")
    return e164, key, ""


async def _send_wa_callmebot(e164: str, text: str) -> tuple[bool, str]:
    _e, key, err = _need_key_and_phone(e164)
    if err:
        return False, err
    return await _http_get(WA_URL,
                           {"phone": e164, "text": text, "apikey": key}, 30.0)


async def _tg_send_text(text: str,
                        timeout_s: float) -> tuple[bool, str]:
    chat = telegram_chat_id()
    if not chat:
        ok, res = await telegram_status()
        _ = (ok, res)
        return False, ("operator must press /start on the bot first "
                       "(no chat id yet)")
    ok, res = await _tg_call("sendMessage",
                             {"chat_id": chat, "text": text}, None, timeout_s)
    if ok:
        return True, "ok"
    return False, str(res.get("description", "sendMessage failed"))[:300]


async def send_whatsapp_text(phone: str, text: str,
                             timeout_s: float = 30.0) -> tuple[bool, str]:
    """Text message to the operator (provider picks the channel).
    Never raises."""
    text = str(text or "").strip()[:4000]
    if not text:
        return False, "empty text"
    prov = provider()
    if prov == "telegram":
        if not str(phone or "").strip():
            return False, "empty phone number"
        return await _tg_send_text(text, timeout_s)
    e164 = normalize_phone(phone)
    if not e164:
        return False, "empty phone number"
    if prov == "callmebot":
        return await _send_wa_callmebot(e164, text)
    return await _gw_post("/wa/send", {"to": e164, "text": text}, timeout_s)


def encode_ogg_opus(wav_f32, sr: int = 24000) -> bytes:
    """float32 mono wav -> ogg/opus bytes (WhatsApp voice-note codec).

    Raises RuntimeError on ffmpeg failure (callers convert to (False, …)).
    """

    pcm = np.ascontiguousarray(np.asarray(wav_f32, dtype=np.float32))
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "f32le", "-ar", str(int(sr)), "-ac", "1", "-i", "pipe:0",
             "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "pipe:1"],
            input=pcm.tobytes(), capture_output=True, timeout=60)
    except Exception as exc:
        raise RuntimeError(f"ffmpeg failed: {exc}"[:200])
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(
            f"ffmpeg failed: {(p.stderr or b'')[:200].decode(errors='replace')}")
    return bytes(p.stdout)


async def send_whatsapp_voice(phone: str, text: str = "",
                              wav=None, sr: int = 24000,
                              ogg_b64: str = "",
                              timeout_s: float = 60.0) -> tuple[bool, str]:
    """Voice note to the operator. Provide ONE of: ogg_b64, wav (+sr),
    or text (synth happens in the caller, e.g. via server /tts/say — this
    module stays weight-free). Never raises."""

    prov = provider()
    if prov == "callmebot":
        return False, ("unsupported: CallMeBot has no voice-message API "
                       "(switch CONTACT_PROVIDER=telegram)")
    if prov == "telegram":
        if not str(phone or "").strip():
            return False, "empty phone number"
        chat = telegram_chat_id()
        if not chat:
            return False, ("operator must press /start on the bot first "
                           "(no chat id yet)")
    else:
        e164 = normalize_phone(phone)
        if not e164:
            return False, "empty phone number"
    try:
        if ogg_b64:
            raw = base64.b64decode(str(ogg_b64))
        elif wav is not None:
            raw = encode_ogg_opus(wav, sr)
        elif str(text or "").strip():
            return False, ("voice synth needs audio: pass wav/ogg_b64 "
                           "(e.g. from server /tts/say)")
        else:
            return False, "empty voice payload (text/wav/ogg_b64)"
    except RuntimeError as exc:
        return False, str(exc)[:200]
    except Exception as exc:
        return False, f"encode failed: {exc}"[:200]
    if prov == "telegram":
        ok, res = await _tg_call(
            "sendVoice", {"chat_id": chat},
            {"voice": ("voice.ogg", raw, "audio/ogg")}, timeout_s)
        if ok:
            return True, "ok"
        return False, str(res.get("description", "sendVoice failed"))[:300]
    b64 = base64.b64encode(raw).decode()
    return await _gw_post("/wa/send-voice", {"to": e164, "ogg_b64": b64},
                          timeout_s)


async def call_and_speak(phone: str, text: str, lang: str = "en",
                         timeout_s: float = 60.0) -> tuple[bool, str]:
    """Phone call leg — DEFERRED (gateway cannot dial PSTN).

    Only fires when CONTACT_CALLMEBOT_API_KEY is set (TTS-call fallback).
    Never raises.
    """
    key = _api_key()
    if not key:
        return False, CALL_DEFERRED
    e164 = normalize_phone(phone)
    if not e164:
        return False, "empty phone number"
    text = str(text or "").strip()[:500]
    if not text:
        return False, "empty text"
    return await _http_get(CALL_URL, {"phone": e164, "text": text,
                                      "apikey": key, "lang": lang or "en"},
                           timeout_s)


async def notify(phone: str, text: str,
                 channels: tuple[str, ...] = ("whatsapp", "call")) -> dict:
    """Fire ``text`` down each channel; ``{channel: (ok, reason)}``."""
    out: dict = {}
    for ch in channels or ():
        if ch == "whatsapp":
            out[ch] = await send_whatsapp_text(phone, text)
        elif ch == "call":
            out[ch] = await call_and_speak(phone, text)
        elif ch in ("voice", "ask"):
            out[ch] = (False, f"use send_whatsapp_{ch}/ask_on_whatsapp "
                              "directly (needs its own args)")
        else:
            out[ch] = (False, f"unknown channel: {ch}"[:100])
    return out


async def ask_on_whatsapp(phone: str, question: str,
                          timeout_s: float = 120.0
                          ) -> tuple[str, str | None, str]:
    """Send ``question`` to the operator, wait for the inbound reply.

    Returns ``(ask_id, answer, status)``; status is ``"ok"``,
    ``"timeout: ..."``, or the send failure reason. The wait resolves via
    :func:`receive_inbound` (real reply through the gateway/poller) or
    :func:`receive_answer` (explicit ask_id, e.g. ``POST /contact/reply``).
    Never raises.
    """
    prov = provider()
    if prov == "telegram":
        route = f"tg:{telegram_chat_id()}" if telegram_chat_id() else ""
        if not str(phone or "").strip():
            return "", None, "empty phone number"
        e164 = ""  # telegram routes on chat id, not E.164
    else:
        e164 = normalize_phone(phone)
        if not e164:
            return "", None, "empty phone number"
        route = e164
    question = str(question or "").strip()[:4000]
    if not question:
        return "", None, "empty question"
    ok, reason = await send_whatsapp_text(phone, question)
    if not ok:
        return "", None, reason
    if prov == "telegram":
        route = f"tg:{telegram_chat_id()}"  # resolved by the send above
    ask_id = f"ask-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    _pending_asks[ask_id] = (fut, route)
    try:
        answer = await asyncio.wait_for(asyncio.shield(fut), timeout_s)
        return ask_id, str(answer)[:2000], "ok"
    except asyncio.TimeoutError:
        return ask_id, None, f"timeout: no reply within {timeout_s:g}s"
    except Exception as exc:
        return ask_id, None, f"wait failed: {exc}"[:200]
    finally:
        _pending_asks.pop(ask_id, None)


def _resolve(ask_id: str, answer: str) -> bool:
    try:
        slot = _pending_asks.get(str(ask_id or ""))
        if slot is None:
            return False
        fut, _phone = slot
        if fut.done():
            return False
        fut.set_result(str(answer or ""))
        return True
    except Exception:
        return False


def receive_answer(ask_id: str, answer: str) -> bool:
    """Resolve a pending :func:`ask_on_whatsapp` wait by ask_id."""
    return _resolve(ask_id, answer)


def receive_inbound(phone: str, text: str) -> bool:
    """Resolve the newest pending ask from this sender.

    ``phone`` is E.164 for WhatsApp or ``tg:<chat_id>`` for Telegram
    (tg: keys match literally, never through E.164 normalization).
    """
    try:
        key = (str(phone or "").strip() if str(phone or "").strip()
               .startswith("tg:") else normalize_phone(phone))
        cands = [k for k, (_f, p) in _pending_asks.items() if p == key]
        if not cands:
            # fallback: any single pending ask (dev / single-operator box)
            cands = list(_pending_asks)
            if len(cands) != 1:
                return False
        return _resolve(sorted(cands)[-1], text)
    except Exception:
        return False
