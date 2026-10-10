"""Creator-bot pairing: PAIRING_KEY -> agent -> chat -> bot token.

One guided in-chat onboarding instead of web-page hopping. The local
backend mints a cryptographically random key; the /tg/connect page shows
a QR encoding ``https://t.me/<CreatorBot>?start=<KEY>``; the user scans,
chats with the Creator Bot (/start KEY -> name -> BotFather guide ->
pastes new bot token), and the backend validates + stores the
association. Tokens never appear in logs or status responses.

Store (gitignored): sessions/tg/pairing.json::

    {"keys": {key: {agent_id, created, expires, status, chat_id,
                    agent_name, fail_reason}},
     "agents": {agent_id: {name, chat_id, bot_token, bot_username,
                           paired_at, pairing_key}}}

Key statuses: new -> started (chat bound) -> named -> done | failed.
Keys are single-use (one chat) and expire after PAIRING_TTL_S (30 min).
"""
import json
import os
import re
import secrets
import time

from src.agent import contacts

__all__ = [
    "PAIRING_TTL_S",
    "new_pairing",
    "get_pairing",
    "validate_pairing",
    "bind_chat",
    "set_name",
    "fail_pairing",
    "finalize_pairing",
    "sanitize_record",
    "creator_token",
    "creator_info",
    "set_creator_token",
    "creator_deeplink",
    "handle_creator_message",
]

PAIRING_TTL_S = 1800

TOKEN_RE = re.compile(r"\d{6,}:[A-Za-z0-9_-]{20,}")


def _tg_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(here, "sessions", "tg")


def _pairing_file() -> str:
    return os.path.join(_tg_dir(), "pairing.json")


def _creator_file() -> str:
    return os.path.join(_tg_dir(), "creator.json")


def _load() -> dict:
    try:
        with open(_pairing_file()) as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                data.setdefault("keys", {})
                data.setdefault("agents", {})
                return data
    except Exception:
        pass
    return {"keys": {}, "agents": {}}


def _save(data: dict) -> bool:
    try:
        os.makedirs(_tg_dir(), exist_ok=True)
        tmp = _pairing_file() + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, _pairing_file())
        return True
    except Exception:
        return False


def _prune(data: dict) -> dict:
    now = time.time()
    data["keys"] = {k: v for k, v in data.get("keys", {}).items()
                    if float(v.get("expires", 0)) > now
                    and v.get("status") != "failed"}
    return data


def new_pairing(agent_id: str = "default") -> tuple[str, int]:
    """Mint a single-use key. Returns (key, ttl_s). Never raises."""
    try:
        agent_id = str(agent_id or "default")[:64]
        key = secrets.token_urlsafe(24)
        now = time.time()
        data = _prune(_load())
        data["keys"][key] = {"agent_id": agent_id, "created": now,
                             "expires": now + PAIRING_TTL_S,
                             "status": "new", "chat_id": "",
                             "agent_name": "", "fail_reason": ""}
        _save(data)
        return key, PAIRING_TTL_S
    except Exception:
        return "", 0


def get_pairing(key: str) -> dict | None:
    try:
        return _load().get("keys", {}).get(str(key or ""))
    except Exception:
        return None


def validate_pairing(key: str) -> dict | None:
    """Live record or None (unknown/expired/failed/done). Never raises."""
    try:
        rec = get_pairing(key)
        if not rec or rec.get("status") in ("done", "failed"):
            return None
        if float(rec.get("expires", 0)) <= time.time():
            return None
        return rec
    except Exception:
        return None


def sanitize_record(rec: dict) -> dict:
    """Status-safe view: no tokens, masked chat id."""
    try:
        chat = str(rec.get("chat_id", "") or "")
        masked = (chat[:3] + "…" + chat[-2:]) if len(chat) > 5 else (
            "set" if chat else "")
        return {"agent_id": rec.get("agent_id", ""),
                "status": rec.get("status", ""),
                "agent_name": rec.get("agent_name", ""),
                "chat": masked,
                "bot": rec.get("bot_username", "")}
    except Exception:
        return {}


def bind_chat(key: str, chat_id: str) -> dict | None:
    """Single-use bind: first chat wins, others rejected. Never raises."""
    try:
        chat_id = str(chat_id or "").strip()
        if not chat_id:
            return None
        data = _load()
        rec = data.get("keys", {}).get(str(key or ""))
        if not rec or rec.get("status") in ("done", "failed"):
            return None
        if float(rec.get("expires", 0)) <= time.time():
            return None
        if rec.get("chat_id") and rec["chat_id"] != chat_id:
            return None
        rec["chat_id"] = chat_id
        if rec.get("status") == "new":
            rec["status"] = "started"
        _save(data)
        return rec
    except Exception:
        return None


def set_name(key: str, name: str) -> dict | None:
    try:
        name = " ".join(str(name or "").split())[:64]
        if not name:
            return None
        data = _load()
        rec = data.get("keys", {}).get(str(key or ""))
        if not rec or rec.get("status") not in ("started", "named"):
            return None
        rec["agent_name"] = name
        rec["status"] = "named"
        _save(data)
        return rec
    except Exception:
        return None


def fail_pairing(key: str, reason: str) -> None:
    try:
        data = _load()
        rec = data.get("keys", {}).get(str(key or ""))
        if rec and rec.get("status") != "done":
            rec["status"] = "failed"
            rec["fail_reason"] = str(reason or "")[:200]
            _save(data)
    except Exception:
        pass


def finalize_pairing(key: str, bot_token: str
                     ) -> tuple[bool, str | dict]:
    """Validate the agent-bot token, store the association, mark done.

    Returns (True, agent_record) or (False, reason). Token validation goes
    through contacts.set_tg_token (live getMe); the token is also recorded
    under agents[agent_id]. Never raises, never logs the token.
    """
    try:
        data = _load()
        rec = data.get("keys", {}).get(str(key or ""))
        if not rec or rec.get("status") not in ("started", "named"):
            return False, "pairing not ready (scan the QR first)"
        m = TOKEN_RE.search(str(bot_token or ""))
        if not m:
            return False, "no bot token found in that message"
        token = m.group(0)
        ok, info = contacts.validate_tg_token_sync(token)
        if not ok:
            return False, info
        username = info.lstrip("@") if info.startswith("@") else ""
        if not contacts._tg_save_bot(token, username):
            return False, "could not persist token"
        agent_id = rec.get("agent_id", "default") or "default"
        agent_name = rec.get("agent_name", "") or agent_id
        contacts.register_agent(agent_id, agent_name, rec.get("chat_id", ""),
                                token, username)
        agent = {"name": rec.get("agent_name", "") or agent_id,
                 "chat_id": rec.get("chat_id", ""),
                 "bot_token": token,
                 "bot_username": username,
                 "paired_at": time.time(),
                 "pairing_key": str(key or "")}
        data["agents"][agent_id] = agent
        rec["status"] = "done"
        rec["bot_username"] = username
        _save(data)
        if not contacts.telegram_chat_id() and rec.get("chat_id"):
            contacts.remember_tg_chat(rec["chat_id"])
        return True, {k: v for k, v in agent.items() if k != "bot_token"}
    except Exception as exc:
        return False, f"finalize failed: {exc}"[:200]


# -- creator bot identity -------------------------------------------------

def creator_token() -> str:
    env = (os.environ.get("CREATOR_BOT_TOKEN") or "").strip()
    if env:
        return env
    try:
        with open(_creator_file()) as fh:
            return str(json.load(fh).get("token", "") or "").strip()
    except Exception:
        return ""


def creator_info() -> dict:
    try:
        with open(_creator_file()) as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def set_creator_token(token: str) -> tuple[bool, str]:
    """Validate via live getMe, persist (no env needed). Never raises."""
    try:
        token = str(token or "").strip()
        if not token or ":" not in token:
            return False, "that doesn't look like a bot token (want 123:ABC…)"
        ok, res = contacts._tg_call_sync("getMe", None, None, 15.0,
                                         token=token)
        if not ok:
            return False, str(res.get("description", "getMe failed"))[:200]
        username = str(res.get("username", "") or "")
        os.makedirs(_tg_dir(), exist_ok=True)
        with open(_creator_file(), "w") as fh:
            json.dump({"token": token, "username": username}, fh)
        os.environ["CREATOR_BOT_TOKEN"] = token
        return True, ("@" + username) if username else "ok"
    except Exception as exc:
        return False, f"save failed: {exc}"[:200]


def creator_deeplink(key: str) -> str:
    username = str(creator_info().get("username", "") or "").strip()
    if not username:
        return ""
    return f"https://t.me/{username}?start={key}"


# -- creator conversation -------------------------------------------------

def _suggest_username(name: str) -> str:
    base = "".join(c.lower() for c in str(name or "") if c.isalnum())
    return ((base[:20] or "voice") + "_agent_bot")[:32]


def _guide_text(name: str) -> str:
    sug = _suggest_username(name)
    return (
        f"Nice — your agent will be called “{name}”.\n\n"
        "Now create its Telegram bot (2 min, one time):\n"
        "1 · Open @BotFather\n"
        "2 · Send /newbot\n"
        f"3 · Name: paste this → {name}\n"
        f"4 · Username: paste this → {sug} (must end in “bot”; "
        "if taken, add digits)\n"
        "5 · BotFather replies with a token like 123:ABC…\n"
        "6 · Paste that token HERE — I validate + wire it instantly.")


def handle_creator_message(chat_id: str, text: str) -> str | None:
    """One private message -> reply text (or None to stay silent).

    Never raises, never echoes tokens back.
    """
    try:
        chat_id = str(chat_id or "").strip()
        text = str(text or "").strip()[:2000]
        if not chat_id or not text:
            return None
        parts = text.split(None, 1)
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")

        if cmd == "/start":
            key = arg.strip()
            rec = validate_pairing(key) if key else None
            if rec is None:
                if not key:
                    return ("👋 Hi! To pair a voice agent, scan the QR on "
                            "your agent's /tg/connect page first, then come "
                            "back here.")
                return ("🔑 That pairing link is unknown, expired, or "
                        "already used — generate a fresh one on the "
                        "/tg/connect page.")
            bound = bind_chat(key, chat_id)
            if bound is None:
                return ("🔒 This pairing link is already bound to another "
                        "chat — generate a fresh one on /tg/connect.")
            if bound.get("status") == "done":
                return "✅ Already paired — your agent is live."
            if bound.get("agent_name"):
                return _guide_text(bound["agent_name"])
            return ("🔗 Paired! What should your voice agent be called? "
                    "(one short name, e.g. Friday)")

        # find this chat's live pairing
        data = _load()
        mine = [k for k, v in data.get("keys", {}).items()
                if v.get("chat_id") == chat_id
                and v.get("status") in ("started", "named")]
        if not mine:
            return ("👋 Scan the pairing QR on your agent's /tg/connect "
                    "page first — then we'll set up your bot here.")
        key = sorted(mine)[-1]
        rec = data["keys"][key]

        # token pasted (early or on time)? finalize immediately.
        if TOKEN_RE.search(text):
            ok, info = finalize_pairing(key, text)
            if ok:
                agent = info if isinstance(info, dict) else {}
                deep = (f"https://t.me/{agent.get('bot_username')}"
                        if agent.get("bot_username") else "")
                nxt = (f"\n\nLast step: open {deep} and press /start — "
                       "then your agent can message you." if deep else
                       "\n\nLast step: open your new bot and press /start.")
                return (f"✅ “{agent.get('name', 'agent')}” is wired!"
                        f"{nxt}")
            return f"❌ {info} — paste the BotFather token again."

        if rec.get("status") == "started":
            named = set_name(key, text)
            if named is None:
                return "❓ Give me a short name for your voice agent."
            return _guide_text(named["agent_name"])

        # status named, no token yet: reprompt with the guide
        return _guide_text(rec.get("agent_name", "agent"))
    except Exception:
        return "⚠️ hiccup — try that again?"
