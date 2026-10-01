"""Shared Telegram long-poll primitives (stdlib only).

Both gates (agent replies, creator pairing) poll Bot API getUpdates the
same way; only the per-message handling differs. No third-party deps so
the gates run even on a bare interpreter.
"""
import json
import time
import urllib.request

__all__ = ["api_call", "run_poll"]


def api_call(token: str, method: str, payload=None,
             timeout: float = 40.0) -> dict:
    """POST-JSON Bot API call. Returns the decoded envelope (or an
    ok:False envelope on transport failure). Never raises."""
    try:
        data = json.dumps(payload or {}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/{method}", data=data,
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as exc:
        return {"ok": False, "description": f"request failed: {exc}"[:200]}


def run_poll(token: str, poll_s: float, on_private_text, log) -> int:
    """Long-poll until KeyboardInterrupt. on_private_text(chat_id, text).

    Returns 0 on clean stop. Never crashes on bad updates; backs off on
    API errors (409 = another poller holds the token).
    """
    me = api_call(token, "getMe", timeout=15.0)
    if not me.get("ok"):
        log("getMe failed:", str(me.get("description", "?"))[:200])
        return 1
    log("bot: @" + str((me.get("result") or {}).get("username", "?")),
        "— polling")
    offset = 0
    backoff = 1.0
    while True:
        try:
            res = api_call(token, "getUpdates",
                           {"offset": offset, "timeout": int(poll_s),
                            "allowed_updates": ["message"]},
                           timeout=poll_s + 15)
            if not res.get("ok"):
                desc = str(res.get("description", ""))[:200]
                log("getUpdates:", desc)
                if "409" in desc or "conflict" in desc.lower():
                    log("another poller holds this token — "
                        "stop the other gate first")
                time.sleep(min(backoff, 30.0))
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            for upd in res.get("result") or []:
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                msg = upd.get("message") or {}
                chat = msg.get("chat") or {}
                if chat.get("type") != "private":
                    continue
                chat_id = str(chat.get("id", ""))
                text = str(msg.get("text", "") or "")[:2000]
                if not chat_id or not text.strip():
                    continue
                try:
                    on_private_text(chat_id, text)
                except Exception as exc:  # handler must not kill the loop
                    log("handler error:", str(exc)[:200])
        except KeyboardInterrupt:
            log("stop")
            return 0
        except Exception as exc:
            log("loop error:", str(exc)[:200])
            time.sleep(min(backoff, 30.0))
            backoff = min(backoff * 2, 30.0)
