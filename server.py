"""voice-agent server: HTTP endpoints + WS /talk over one :class:`VoiceAgent`.

- ``GET /health`` — liveness, never builds the agent.
- ``GET /metrics`` — uptime, turn stats, per-node event counts.
- ``POST /contact/reply`` — inbound-reply landing zone for ask_on_whatsapp.
- ``WS /talk`` — two-way socket: PCM audio in, per-node streams out.

Minimal ``/talk`` protocol (JSON text unless noted)::

    client -> server: {"type": "config", "sr": 16000,
                       "session_id": "...", "end_silence_s": 2.0,
                       "partial_s": 0.7}
                       (partial_s > 0 enables live STT partials, 0 = off)
    client -> server: {"type": "audio", "pcm_b64": "<pcm16 mono>", "sr": 16000}
    client -> server: <binary pcm16 mono frames>  (same as an audio chunk)
    client -> server: {"type": "commit"}   run a turn on the buffer now
    client -> server: {"type": "reset"}    drop the buffer + VAD state
    client -> server: {"type": "close"}    end the session

    server -> client: {"event": "ready", "session_id": ..., "sr": 16000,
                       "nodes": ["vad", "stt", "llm", "tts"]}
    server -> client: {"event": "node", "node": "vad"|"stt"|"llm"|"tts",
                       "kind": ..., "data": {...}}
                      (tts audio arrives as data.wav_b64 + sr + sentence;
                       live STT partials arrive as node=stt kind=partial
                       while speech is active, final as kind=text)
    server -> client: {"event": "done", "summary": {text, reply, node_s,
                       ttfa_s, total_s, session_id}}
    server -> client: {"event": "error", "message": ...}

Endpointing: every chunk is VAD-scored; after speech plus
``end_silence_s`` trailing silence the buffered turn auto-commits, so a
client can just stream mic frames and read back node streams.
"""
import asyncio
import base64
import threading
import time

import numpy as np
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

__all__ = ["create_app", "app", "router", "browser_router"]

router = APIRouter()
browser_router = APIRouter()

CLIENT_SR = 16000
END_SILENCE_S = 2.0
MAX_BUFFER_S = 60.0
# Live STT partials: re-decode the trailing window every partial_s of new
# audio while speech is active. Never overlaps (one in flight max), never
# blocks the socket, never raises into the loop.
PARTIAL_MIN_S = 1.0
PARTIAL_WINDOW_S = 15.0

_t_boot = time.time()
_metrics = {"turns_started": 0, "turns_done": 0, "turn_errors": 0,
            "turn_total_s": 0.0, "events": {}}
_metrics_lock = threading.Lock()

_agent = None


def _record_event(node: str):
    with _metrics_lock:
        ev = _metrics["events"]
        ev[node] = ev.get(node, 0) + 1


def _record_turn(dt: float, err: bool = False):
    with _metrics_lock:
        _metrics["turns_started"] += 1
        if err:
            _metrics["turn_errors"] += 1
        else:
            _metrics["turns_done"] += 1
            _metrics["turn_total_s"] += dt


def _engine_stats(agent) -> dict | None:
    """Paged engine stats off the LLM leg (it owns the engine). Never raises."""
    try:
        llm = getattr(agent, "llm", None)
        if llm is None:
            return None
        eng = llm._paged_engine()  # type: ignore
        if eng is None:
            return None
        return eng.stats()  # type: ignore
    except Exception:
        return None


def _paged_requested(agent) -> bool:
    """Whether the LLM leg was configured for the paged engine."""
    try:
        return bool(getattr(getattr(agent, "llm", None), "config", None)
                    and getattr(getattr(agent, "llm", None).config,
                                "use_paged", False))
    except Exception:
        return False


def get_agent():
    """Process-wide :class:`VoiceAgent`, built on first use.

    Two brains by default (src/agent/delegate.py): a Qwen3-0.6B front leg
    that talks to the user and is handed no tools at all — it only answers
    yes/no on "does this need the worker?" — and the Bonsai worker leg
    behind it carrying the full agentic harness.
    Placed so the worker keeps the GPU: STT/TTS on GPU, worker offloaded
    by availability, front as small as the config allows.

    Env overrides (same pattern as VOICE_TEXT_ONLY in bridge.py):
    ``VOICE_LLM_BACKEND`` (qwen or bonsai),
    ``VOICE_LLM_MODEL`` (HF id for GPU-fused backends),
    ``VOICE_FAST_VOICE=1`` (one direct generate per turn, no agent loop),
    ``VOICE_DELEGATE=0`` (single brain: the llm leg answers everything),
    ``VOICE_DELEGATE_CONFIG`` (path to delegate.yaml).
    """
    global _agent
    if _agent is None:
        import os

        from src.main import VoiceAgent, VoiceAgentConfig

        backend = os.environ.get("VOICE_LLM_BACKEND", "").strip()
        model = os.environ.get("VOICE_LLM_MODEL", "").strip()
        fast = os.environ.get("VOICE_FAST_VOICE", "").strip() == "1"
        delegate = os.environ.get("VOICE_DELEGATE", "").strip()
        dpath = os.environ.get("VOICE_DELEGATE_CONFIG", "").strip() or None
        tts_voice = os.environ.get("VOICE_TTS_VOICE", "").strip()
        from dataclasses import replace

        from src.models.llm import LlmConfig

        # Two brains by default: this endpoint IS the product, so the
        # front/worker split is the default here even though the library
        # default is off (see VoiceAgentConfig.delegate).
        cfg = VoiceAgentConfig(
            delegate=delegate not in ("0", "false", "no"),
            delegate_config=dpath)
        if backend or model:
            cfg = replace(cfg, llm=LlmConfig(
                backend=backend or LlmConfig.backend,
                model=model or LlmConfig.model))
        if fast:
            cfg = replace(cfg, fast_voice=True)
        if tts_voice:
            from src.models.tts import TtsConfig

            cfg = replace(cfg, tts=TtsConfig(
                voice=tts_voice, lang=cfg.tts.lang,
                sample_rate=cfg.tts.sample_rate, speed=cfg.tts.speed,
                device=cfg.tts.device))
        _agent = VoiceAgent(cfg)
    return _agent


def _event_frame(event) -> dict:
    """AgentEvent -> minimal WS frame (wav bytes become b64)."""
    node, kind = event.node, event.kind
    if node == "turn":
        return {"event": "done", "summary": dict(event.data or {})}
    data = dict(event.data or {})
    wav = data.pop("wav", None)
    if wav is not None:
        arr = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
        data["wav_b64"] = base64.b64encode(arr.tobytes()).decode()
    return {"event": "node", "node": node, "kind": kind, "data": data}


def _pcm16_to_float(raw: bytes) -> np.ndarray:
    return (np.frombuffer(bytes(raw), dtype="<i2").astype("float32")
            / 32768.0)


@router.get("/health")
def health():
    # Liveness-safe: never builds the agent, never throws. Always returns
    # the full spec shape {ok, uptime_s, agent_loaded, missing, nodes,
    # sessions, vram_mb, engine} — with defaults until the agent warms.
    try:
        loaded = bool(_agent is not None and getattr(_agent, "warmed", False))
    except Exception:
        loaded = False
    out: dict = {"ok": True, "uptime_s": time.time() - _t_boot,
                 "agent_loaded": loaded,
                 "missing": [],
                 "nodes": ["vad", "stt", "llm", "tts"],
                 "sessions": {"sessions": 0, "turns": 0, "max_turns": 20,
                              "max_age_s": 1800.0},
                 "vram_mb": 0.0,
                 "engine": None}
    if _agent is not None:
        try:
            out["missing"] = list(getattr(_agent, "missing", []) or [])
        except Exception:
            pass
        try:
            out["sessions"] = _agent.sessions.stats()
        except Exception:
            pass
        # two-brain routing state: which leg answers, and whether the front
        # brain is actually live. A front leg that failed to warm is the
        # single most useful thing to see here — every turn silently falls
        # back to the worker without it.
        try:
            dcfg = getattr(_agent, "delegate_cfg", None)
            if dcfg is not None:
                front = getattr(_agent, "front_llm", None)
                fcfg = getattr(front, "config", None)
                worker = getattr(_agent, "llm", None)
                wcfg = getattr(worker, "config", None)
                out["delegate"] = {
                    "enabled": True,
                    "placement": str(getattr(dcfg, "placement", "")),
                    "backstop": bool(getattr(dcfg, "backstop", False)),
                    "front": {
                        "model": getattr(fcfg, "model", None),
                        "backend": getattr(fcfg, "backend", None),
                        "device": getattr(fcfg, "device", None),
                        "built": front is not None,
                    },
                    "worker": {
                        "model": getattr(wcfg, "model", None),
                        "backend": getattr(wcfg, "backend", None),
                    },
                    # Where the 0.6B and the regex scorer disagreed, so
                    # routing accuracy can be judged on live traffic.
                    "skew": dict(getattr(_agent, "_route_skew", {}) or {}),
                }
            else:
                out["delegate"] = {"enabled": False}
        except Exception:
            pass
        # paged engine stats — proves inference engine is live for LLM
        try:
            stats = _engine_stats(_agent)
            if stats is not None:
                out["engine"] = {
                    "active": True,
                    "runner": stats.get("runner"),
                    "device": stats.get("device"),
                    "steps": stats.get("steps"),
                    "kv_cache": stats.get("kv_cache"),
                    "scheduler": stats.get("scheduler"),
                    "prefix_cache": stats.get("prefix_cache"),
                    "cuda_graph": stats.get("cuda_graph"),
                }
            else:
                # engine configured but not yet warmed, or non-paged mode
                out["engine"] = {"active": False,
                                 "paged_requested": _paged_requested(_agent)}
        except Exception:
            pass
    try:
        from src.models.runtime import max_allocated_mb

        out["vram_mb"] = float(max_allocated_mb())
    except Exception:
        pass
    return out


@router.get("/vad", include_in_schema=False)
def vad_page():
    """Single-file VAD -> ASR live test page (mic over /talk WS)."""
    import os

    from fastapi.responses import FileResponse

    here = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(here, "web", "vad_asr.html"))


@router.get("/latency", include_in_schema=False)
def latency_page():
    """Speech-to-speech page with end-to-end latency readout."""
    import os

    from fastapi.responses import FileResponse

    here = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(here, "web", "latency.html"))


@router.get("/metrics")
def metrics():
    with _metrics_lock:
        snap = {"turns_started": _metrics["turns_started"],
                "turns_done": _metrics["turns_done"],
                "turn_errors": _metrics["turn_errors"],
                "events": dict(_metrics["events"]),
                "turn_total_s": _metrics["turn_total_s"]}
    done = snap["turns_done"]
    events = {"vad": 0, "stt": 0, "llm": 0, "tts": 0, "turn": 0}
    events.update(snap["events"])
    out = {
        "uptime_s": time.time() - _t_boot,
        "turns": {
            "started": snap["turns_started"],
            "done": done,
            "errors": snap["turn_errors"],
            "mean_s": (snap["turn_total_s"] / done) if done else 0.0,
        },
        "events": events,
    }
    # enrich with paged engine counters if live
    try:
        estats = _engine_stats(_agent)
        if estats is not None:
            out["engine"] = {
                "steps": estats.get("steps"),
                "total_tokens": estats.get("total_tokens"),
                "runner": estats.get("runner"),
                "kv_cache": estats.get("kv_cache"),
                "prefix_cache": estats.get("prefix_cache"),
                "cuda_graph": estats.get("cuda_graph"),
            }
    except Exception:
        pass
    return out


@router.post("/contact/reply")
async def contact_reply(payload: dict):
    """Inbound-reply landing zone for ask_on_whatsapp (CallMeBot cannot
    deliver WhatsApp replies, so answers arrive here instead).

    Request: {ask_id, answer}. Response: {ok, ask_id}.
    """
    from src.agent import contacts

    ask_id = str((payload or {}).get("ask_id", "") or "")[:128]
    answer = str((payload or {}).get("answer", "") or "")[:2000]
    if not ask_id:
        return {"ok": False, "error": "empty ask_id"}
    return {"ok": bool(contacts.receive_answer(ask_id, answer)),
            "ask_id": ask_id}


@router.post("/contact/inbound")
async def contact_inbound(payload: dict):
    """Gateway callback for real WhatsApp replies: {phone, text}.

    Resolves the newest pending ask_on_whatsapp from that phone.
    Response: {ok}.
    """
    from src.agent import contacts

    phone = str((payload or {}).get("phone", "") or "")[:32]
    text = str((payload or {}).get("text", "") or "")[:2000]
    if not phone or not text.strip():
        return {"ok": False, "error": "empty phone/text"}
    if phone.startswith("tg:"):
        contacts.remember_tg_chat(phone[3:])
    return {"ok": bool(contacts.receive_inbound(phone, text))}


@router.get("/wa/connect", include_in_schema=False)
def wa_connect_page():
    """One-scan WhatsApp linking page (QR from wa-gate)."""
    import os

    from fastapi.responses import FileResponse

    here = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(here, "web", "wa_connect.html"))


@router.get("/wa/qr")
def wa_qr():
    """Proxy wa-gate QR/health so the browser stays same-origin."""
    import os

    try:
        import httpx

        base = (os.environ.get("CONTACT_GATEWAY_URL")
                or "http://127.0.0.1:8100").rstrip("/")
        r = httpx.get(base + "/wa/health", timeout=5.0)
        health = r.json() if r.status_code == 200 else {}
        if health.get("linked"):
            return {"linked": True, "phone": health.get("phone", "")}
        q = httpx.get(base + "/wa/qr", timeout=10.0)
        qd = q.json() if q.status_code == 200 else {}
        return {"linked": False,
                "qr_data_url": qd.get("qr_data_url"),
                "waiting": qd.get("waiting", qd.get("qr_data_url") is None)}
    except Exception as exc:
        return {"linked": False, "error": f"gateway down: {exc}"[:200]}


@router.get("/tg/connect", include_in_schema=False)
def tg_connect_page():
    """Telegram linking page: create bot, /start once, linked."""
    import os

    from fastapi.responses import FileResponse

    here = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(here, "web", "tg_connect.html"))


@router.get("/tg/status")
async def tg_status():
    """Telegram bot link state + creator gate state (never includes tokens).
    {linked, chat_id?, bot?|err, creator: {linked, bot?|err}}"""
    import time as _t

    from src.agent import contacts, pairing

    t0 = _t.perf_counter()
    try:
        out = await contacts.telegram_status()
    except Exception as exc:  # never 500
        _record_turn(_t.perf_counter() - t0, err=True)
        out = {"linked": False, "err": str(exc)[:200]}
    try:
        ctok = pairing.creator_token()
        cinfo = pairing.creator_info()
        cbot = str(cinfo.get("username", "") or "")
        out["creator"] = ({"linked": True, "bot": "@" + cbot} if ctok and cbot
                          else {"linked": bool(ctok),
                                "err": "creator token not set"} if not ctok
                          else {"linked": False,
                                "err": "creator username unknown"})
    except Exception:
        out["creator"] = {"linked": False, "err": "creator check failed"}
    return out


@router.post("/tg/creator-token")
async def tg_creator_token(payload: dict):
    """Save the Creator Bot token (validates via getMe first).

    Request: {token}. Response: {ok, bot} or {ok: false, error}.
    One-time setup; then POST /tg/pairing/new mints guided-setup QRs.
    """
    from src.agent import pairing

    token = str((payload or {}).get("token", "") or "")[:200]
    if not token.strip():
        return {"ok": False, "error": "empty token"}
    ok, info = pairing.set_creator_token(token)
    if not ok:
        return {"ok": False, "error": info}
    return {"ok": True, "bot": info}


@router.post("/tg/pairing/new")
async def tg_pairing_new(payload: dict):
    """Mint a guided-setup pairing: QR -> Creator Bot chat -> bot token.

    Request: {agent_id?}. Response: {ok, pairing_key, qr_data_url,
    deep_link, expires_in_s} or {ok: false, error}.
    """
    from src.agent import pairing

    agent_id = str((payload or {}).get("agent_id", "") or "default")[:64]
    deep_user = pairing.creator_info().get("username", "")
    if not pairing.creator_token() or not deep_user:
        return {"ok": False,
                "error": "creator bot not set (POST /tg/creator-token first)"}
    key, ttl = pairing.new_pairing(agent_id)
    if not key:
        return {"ok": False, "error": "could not mint pairing key"}
    link = pairing.creator_deeplink(key)
    try:
        import base64
        import io

        import qrcode

        img = qrcode.make(link)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        qr = ("data:image/png;base64,"
              + base64.b64encode(buf.getvalue()).decode())
    except Exception as exc:
        return {"ok": False, "error": f"qr failed: {exc}"[:200]}
    return {"ok": True, "pairing_key": key, "qr_data_url": qr,
            "deep_link": link, "expires_in_s": ttl}


@router.get("/tg/pairing/status")
async def tg_pairing_status(key: str = ""):
    """Sanitized pairing state (tokens never leave the backend)."""
    from src.agent import pairing

    rec = pairing.get_pairing(key)
    if rec is None:
        return {"ok": False, "error": "unknown pairing key"}
    live = pairing.validate_pairing(key) is not None
    out = pairing.sanitize_record(rec)
    out.update({"ok": True, "live": live})
    return out


@router.post("/tg/token")
async def tg_token(payload: dict):
    """Save a BotFather token (validates via getMe first).

    Request: {token}. Response: {ok, bot} or {ok: false, error}.
    Stored under sessions/tg/ (gitignored), live immediately.
    """
    from src.agent import contacts

    token = str((payload or {}).get("token", "") or "")[:200]
    if not token.strip():
        return {"ok": False, "error": "empty token"}
    ok, info = await contacts.set_tg_token(token)
    if not ok:
        return {"ok": False, "error": info}
    return {"ok": True, "bot": info}


@router.get("/tg/qr")
def tg_qr():
    """Deep-link QR: scan with the phone camera → opens the bot chat.

    Response: {linked, qr_data_url?, bot?} (linked skips the QR).
    """
    import asyncio as _aio

    from src.agent import contacts

    try:
        st = _aio.run(contacts.telegram_status())
    except Exception as exc:
        return {"linked": False, "error": str(exc)[:200]}
    if st.get("linked"):
        return {"linked": True, "bot": st.get("bot", "")}
    ok, qr = contacts.tg_deeplink_qr()
    if not ok:
        return {"linked": False, "error": qr,
                "need": "token" if "token" in qr else "start"}
    return {"linked": False, "qr_data_url": qr,
            "bot": st.get("bot", "")}


def prefill_messages(agent, sid, text):
    """Prompt sharing the real turn's prefix, for KV priming.

    Same builder the worker decodes with (system + history + CWD
    trailer), only the tail is the still-growing partial. The decode
    that follows reuses the matched prefix from the leg's cache, so
    only genuinely new words cost prompt time. None when unusable —
    prefill is best-effort and must never break a turn.
    """
    try:
        llm = getattr(agent, "llm", None)
        if llm is None or not str(text or "").strip():
            return None
        try:
            hist = agent.sessions.history(sid) if sid else []
        except Exception:
            hist = []
        return llm.messages_for_terminal(str(text)[:600], hist, ".", "")
    except Exception:
        return None


async def prime_prefill(llm, msgs):
    """Process one prefill prompt without generating; return prompt ms.

    Fused legs fall back to encode() (client template + tokenizer warm;
    paged prefix blocks prime on encode). Single-slot CPU sidecars are
    SKIPPED outright: a prime costs seconds of prompt processing there
    (measured 2.4s), longer than the pause window, and it would hold the
    server's only slot while the real turn waits. None on any failure
    or skip — the turn decodes unprimed, exactly as before.
    """
    try:
        leg = llm._backend() if hasattr(llm, "_backend") else None
        if (leg is not None and hasattr(leg, "_payload")
                and hasattr(leg, "_post")):
            return None
        await llm.encode(msgs)
        return 0.0
    except Exception:
        return None


@router.websocket("/talk")
async def talk(ws: WebSocket):
    await ws.accept()
    from engine import new_session_id

    agent = get_agent()
    sid = ws.query_params.get("session_id") or new_session_id()
    client_sr = CLIENT_SR
    end_silence_s = END_SILENCE_S
    partial_s = 0.0  # <= 0 disables live partials; config may set cadence
    buf = np.zeros(0, dtype=np.float32)
    speech_seen = False
    trailing_sil = 0.0
    last_vad = None
    last_partial = ""
    last_partial_at = 0.0
    partial_busy = False
    turn_seq = 0
    turn_task = None
    prefill_task = None
    try:
        agent.vad.reset()
    except Exception:
        pass  # fresh socket starts with clean VAD state, always
    await ws.send_json({"event": "ready", "session_id": sid, "sr": 16000,
                        "nodes": ["vad", "stt", "llm", "tts"]})

    vad = getattr(agent, "vad", None)
    stt = getattr(agent, "stt", None)

    def reset_partials():
        """New turn/reset: in-flight partials go stale, cadence restarts."""
        nonlocal last_partial, last_partial_at, turn_seq
        turn_seq += 1
        last_partial, last_partial_at = "", 0.0
        prefill_abort()

    def prefill_abort():
        """Drop the in-flight prime; its server-side KV stays cached."""
        nonlocal prefill_task
        task, prefill_task = prefill_task, None
        if task is not None and not task.done():
            task.cancel()

    def prefill_prime(text):
        """Prime the decode leg with the growing transcript, if idle.

        Cancels the previous prime (its KV is still useful server-side)
        and starts one for the latest words. Skipped mid-turn: decode
        owns the leg then. Fire-and-forget; completion reports ms.
        """
        nonlocal prefill_task
        prefill_abort()
        if turn_running() or not str(text or "").strip():
            return
        llm = getattr(agent, "llm", None)
        if llm is None:
            return
        msgs = prefill_messages(agent, sid, text)
        if not msgs:
            return
        seq = turn_seq

        async def _prime():
            ms = await prime_prefill(llm, msgs)
            if ms is None or seq != turn_seq or turn_running():
                return  # superseded, failed, or turn started: stay silent
            _record_event("llm")
            try:
                await ws.send_json({"event": "node", "node": "llm",
                                    "kind": "prefill",
                                    "data": {"prompt_ms": round(ms, 1)}})
            except Exception:
                pass  # disconnect races must never kill the task

        prefill_task = asyncio.create_task(_prime())

    def maybe_partial():
        """Launch one rolling-window transcribe; never overlaps, never raises.

        Re-decodes the trailing window from scratch each time, so each
        partial is self-consistent (continuity comes from overlapping
        audio, no decoder surgery needed). The loop never awaits it.
        """
        nonlocal partial_busy, last_partial_at
        if (partial_s <= 0 or partial_busy or stt is None
                or not speech_seen or len(buf) == 0):
            return
        buf_s = len(buf) / 16000.0
        if buf_s < PARTIAL_MIN_S or buf_s - last_partial_at < partial_s:
            return
        window_n = int(PARTIAL_WINDOW_S * 16000)
        snap = buf[-window_n:].copy() if len(buf) > window_n else buf.copy()
        snap_s = buf_s
        seq = turn_seq
        last_partial_at = buf_s
        partial_busy = True

        async def _run():
            nonlocal partial_busy, last_partial
            try:
                res = await stt.transcribe(snap, 16000)
                text = str(res.text or "").strip()
            except Exception:
                text = ""
            finally:
                partial_busy = False
            if seq != turn_seq or not text or text == last_partial:
                return  # stale turn, empty, or unchanged: stay silent
            last_partial = text
            _record_event("stt")
            try:
                await ws.send_json({"event": "node", "node": "stt",
                                    "kind": "partial",
                                    "data": {"text": text,
                                             "buffer_s": round(snap_s, 2)}})
            except Exception:
                pass  # disconnect races must never kill the task
            # Streaming prefill: the new words prime the decode leg
            # while the user keeps talking (see prefill_prime).
            prefill_prime(text)
            # Prefill: the decode that follows re-encodes this same prefix
            # (history + trailer get prepended at commit, the tail matches),
            # so tokenizing it now warms the encode path and primes the
            # paged prefix cache while the user is still speaking.
            # Fire-and-forget, never raises, skipped mid-turn.
            if not turn_running():
                async def _prefill():
                    try:
                        llm = getattr(agent, "llm", None)
                        if llm is not None:
                            await llm.encode(
                                [{"role": "user",
                                  "content": text[:600]}])
                    except Exception:
                        pass  # prefill is best-effort, never a turn
                asyncio.create_task(_prefill())

        asyncio.create_task(_run())

    async def score(chunk: np.ndarray) -> bool:
        if vad is None or len(chunk) == 0:
            return False
        try:
            return bool(await vad.active(chunk, 16000))
        except Exception:
            return False

    async def emit_vad(speech: bool):
        nonlocal last_vad
        if speech != last_vad:
            last_vad = speech
            _record_event("vad")
            await ws.send_json({"event": "node", "node": "vad",
                                "kind": "speech",
                                "data": {"speech": bool(speech),
                                         "buffer_s": len(buf) / 16000.0}})

    def turn_running():
        return turn_task is not None and not turn_task.done()

    def launch_turn(audio: np.ndarray):
        """Run a turn in the background so the mic loop keeps scoring.

        Returns False when a turn is already running (caller reports it).
        """
        nonlocal turn_task
        if turn_running():
            return False

        async def _wrap():
            try:
                await run_turn(audio)
            finally:
                pass  # turn_running() reads task.done(); nothing to clear

        turn_task = asyncio.create_task(_wrap())
        return True

    async def barge_in():
        """Speech started mid-turn: kill the turn, stop client playback.

        The user's interruption stays in the buffer (it is the next turn's
        input); only the stale turn dies. In-flight partials go stale via
        the sequence bump.
        """
        nonlocal turn_task
        task, turn_task = turn_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        reset_partials()
        _record_event("tts")
        try:
            await ws.send_json({"event": "barge",
                                "reason": "speech-during-playback"})
        except Exception:
            pass  # disconnect races must never kill the task

    async def run_turn(audio: np.ndarray):
        t0 = time.perf_counter()
        try:
            async for event in agent(audio, 16000, sid):
                _record_event(event.node)
                await ws.send_json(_event_frame(event))
        except (ValueError, TimeoutError) as exc:
            _record_turn(time.perf_counter() - t0, err=True)
            await ws.send_json({"event": "error", "message": str(exc)[:300]})
            return
        except Exception as exc:  # noqa: BLE001 - never kill the socket
            try:
                from src.models.runtime import MemoryBudgetExceeded

                if isinstance(exc, MemoryBudgetExceeded):
                    _record_turn(time.perf_counter() - t0, err=True)
                    await ws.send_json({"event": "error",
                                        "message": str(exc)[:300]})
                    return
            except Exception:
                pass
            _record_turn(time.perf_counter() - t0, err=True)
            await ws.send_json({"event": "error", "message": "engine error"})
            return
        _record_turn(time.perf_counter() - t0)
        try:
            agent.vad.reset()
        except Exception:
            pass

    def push(chunk: np.ndarray, sr: int):
        """Append one chunk; returns (speech, endpoint_now)."""
        nonlocal buf, speech_seen, trailing_sil
        if sr != 16000 and len(chunk):
            from engine.audio import resample

            chunk = resample(chunk, sr, 16000)
        buf = np.concatenate([buf, chunk]) if len(buf) else chunk
        if len(buf) / 16000.0 > MAX_BUFFER_S:
            raise ValueError("buffer exceeds 60s; send commit or reset")
        return len(chunk) / 16000.0

    while True:
        try:
            msg = await ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            # RuntimeError: starlette raises 'Cannot call receive once a
            # disconnect message has been received' on client-close races.
            prefill_abort()
            break
        data = msg.get("bytes")
        chunk, sr = None, client_sr
        if data is not None:
            chunk = _pcm16_to_float(bytes(data))
        else:
            text = msg.get("text")
            if text == "close":
                break
            if not text:
                continue
            try:
                import json as _json

                cmd = _json.loads(text)
            except Exception:
                continue
            if not isinstance(cmd, dict):
                continue
            kind = cmd.get("type")
            if kind == "config":
                try:
                    client_sr = int(cmd.get("sr", client_sr))
                    end_silence_s = float(cmd.get("end_silence_s",
                                                  end_silence_s))
                    partial_s = max(0.0, float(cmd.get("partial_s",
                                                       partial_s)))
                    if cmd.get("session_id"):
                        sid = str(cmd["session_id"])[:64]
                except Exception:
                    pass
                await ws.send_json({"event": "ready", "session_id": sid,
                                    "sr": 16000,
                                    "nodes": ["vad", "stt", "llm", "tts"]})
                continue
            if kind == "reset":
                buf = np.zeros(0, dtype=np.float32)
                speech_seen, trailing_sil = False, 0.0
                reset_partials()
                try:
                    agent.vad.reset()
                except Exception:
                    pass
                await ws.send_json({"event": "node", "node": "vad",
                                    "kind": "reset",
                                    "data": {"buffer_s": 0.0}})
                continue
            if kind == "close":
                prefill_abort()
                break
            if kind == "audio":
                try:
                    raw = base64.b64decode(cmd.get("pcm_b64", ""))
                except Exception:
                    await ws.send_json({"event": "error",
                                        "message": "bad pcm_b64"})
                    continue
                chunk, sr = _pcm16_to_float(raw), int(cmd.get("sr", client_sr))
            elif kind == "commit":
                if len(buf) == 0:
                    await ws.send_json({"event": "error",
                                        "message": "empty buffer"})
                    continue
                if turn_running():
                    await ws.send_json({"event": "error",
                                        "message": "turn already running"})
                    continue
                audio, buf = buf, np.zeros(0, dtype=np.float32)
                speech_seen, trailing_sil = False, 0.0
                reset_partials()
                launch_turn(audio)
                continue
            else:
                continue
        if chunk is None:
            continue
        try:
            dur = push(chunk, sr)
        except ValueError as exc:
            await ws.send_json({"event": "error", "message": str(exc)[:200]})
            continue
        speech = await score(chunk)
        await emit_vad(speech)
        if speech:
            speech_seen, trailing_sil = True, 0.0
            # Barge-in: the user started talking over the reply. Kill the
            # stale turn first; this chunk stays buffered as the next turn.
            if turn_running():
                await barge_in()
        else:
            trailing_sil += dur
        if speech_seen:
            maybe_partial()
        if (speech_seen and trailing_sil >= end_silence_s
                and not turn_running()):
            audio, buf = buf, np.zeros(0, dtype=np.float32)
            speech_seen, trailing_sil = False, 0.0
            reset_partials()
            launch_turn(audio)


@browser_router.get("/browser/tools")
def browser_tools():
    """Tool schema the Chrome extension uses to snapshot + validate."""
    from src.agent.prompts import TERMINAL_PREAMBLE
    from src.tools.terminal import ALLOWED_OPS

    agent = get_agent()
    try:
        model = getattr(getattr(agent, "llm", None), "config", None).model
    except Exception:
        model = "single-local-model"
    return {
        "ops": list(ALLOWED_OPS),
        "preamble": TERMINAL_PREAMBLE,
        "snapshot": [{"ref": 1, "role": "button", "name": "Log in"}],
        "action_example": {"action": "click", "ref": 1},
        "model": str(model),
    }


@browser_router.post("/browser/act")
async def browser_act(payload: dict):
    """One agent turn with browser context: chat text back.

    Request: {text, url?, snapshot?:[{ref,role,name}], observation?,
              session_id?}
    Response: {kind: chat, reply, sensitive, raw}
    Runs on the SAME unified VoiceAgent as voice + terminal (the old
    browser sidecar modules no longer exist).
    """
    import time as _t

    t0 = _t.perf_counter()
    text = str((payload or {}).get("text", "") or "")[:2000]
    url = str((payload or {}).get("url", "") or "")[:2000]
    observation = str((payload or {}).get("observation", "") or "")[:1000]
    snapshot = (payload or {}).get("snapshot") or []
    sid = str((payload or {}).get("session_id", "") or "")[:64] or None
    if not text.strip():
        return {"kind": "error", "message": "empty text"}
    if not isinstance(snapshot, list):
        return {"kind": "error", "message": "snapshot must be a list"}
    snapshot = snapshot[:80]
    agent = get_agent()
    refs = " ".join(
        f"[{n.get('ref')}:{n.get('role')}:{str(n.get('name', ''))[:40]}]"
        for n in snapshot if isinstance(n, dict))[:1500]
    prompt = str(text)
    if url:
        prompt += f"\nURL: {url}"
    if refs:
        prompt += f"\nPage: {refs}"
    if observation:
        prompt += "\nLast result: " + observation
    try:
        reply = ""
        async for event in agent.run_text(prompt, session_id=sid):
            if event.kind == "chat":
                reply = str((event.data or {}).get("reply", "") or "")
        _record_event("browser")
        _record_turn(_t.perf_counter() - t0)
        return {"kind": "chat", "reply": reply, "sensitive": False,
                "raw": reply}
    except Exception as exc:  # never 500 — extension loops on error
        _record_turn(_t.perf_counter() - t0, err=True)
        return {"kind": "error", "message": str(exc)[:300]}


@browser_router.post("/tts/say")
async def tts_say(payload: dict):
    """Speak text with the server Kokoro voice (for the visualizer pill).

    Request: {text} (cap 500 chars). Response: {kind: audio, wav_b64
    (float32 LE b64, same encoding as /talk tts frames), sr, sentence}.
    The extension plays it through WebAudio + AnalyserNode so the agent
    voice renders in the pill visualizer instead of speechSynthesis.
    """
    import time as _t

    t0 = _t.perf_counter()
    text = str((payload or {}).get("text", "") or "")[:500]
    if not text.strip():
        return {"kind": "error", "message": "empty text"}
    agent = get_agent()
    tts = getattr(agent, "tts", None)
    if tts is None or not hasattr(tts, "speak"):
        return {"kind": "error", "message": "tts unavailable"}
    try:
        out = await tts.speak(text)
        _record_event("tts")
        _record_turn(_t.perf_counter() - t0)
        arr = np.ascontiguousarray(np.asarray(out.wav, dtype=np.float32))
        return {"kind": "audio",
                "wav_b64": base64.b64encode(arr.tobytes()).decode(),
                "sr": int(out.sample_rate),
                "sentence": str(out.sentence or text)[:500]}
    except Exception as exc:  # never 500
        _record_turn(_t.perf_counter() - t0, err=True)
        return {"kind": "error", "message": str(exc)[:300]}


# FastAPI's OpenAPI generator only documents HTTP routes, so the WS
# route is invisible in /docs and /openapi.json unless we declare it
# by hand (documented as GET: a WebSocket handshake IS an HTTP upgrade).
_TALK_OPENAPI = {
    "get": {
        "summary": "Two-way voice talk (WebSocket)",
        "description": (
            "Bidirectional socket: PCM16 audio chunks in, per-node "
            "streams out. Connect at ws://host:8003/talk "
            "(optional ?session_id=).\n\n"
            "Client -> server: "
            '{"type":"config","sr","session_id","end_silence_s",'
            '"partial_s"} | '
            '{"type":"audio","pcm_b64","sr"} | binary PCM16 mono frames | '
            '{"type":"commit"} | {"type":"reset"} | {"type":"close"}.\n\n'
            "Server -> client: "
            '{"event":"ready","session_id","sr","nodes"} | '
            '{"event":"node","node":"vad"|"stt"|"llm"|"tts",'
            '"kind","data"} (tts audio arrives as data.wav_b64 + sr + '
            "sentence; live STT partials arrive mid-speech as "
            'node=stt kind=partial when config partial_s > 0, final as '
            "kind=text) | "
            '{"event":"done","summary":{text,reply,node_s,ttfa_s,total_s,'
            "session_id}} | "
            '{"event":"error","message"}.'
        ),
        "responses": {
            "101": {"description": "Switching Protocols (WebSocket)"},
        },
    }
}


def create_app(agent=None):
    """Build the voice + browser app; inject an agent (tests) or warm lazily.

    ``router`` keeps /health, /metrics, /talk (voice loop) + /vad (test page).
    ``browser_router`` adds /browser/tools + /browser/act (extension).
    """
    from fastapi import FastAPI

    global _agent
    if agent is not None:
        _agent = agent
    app = FastAPI(title="voice-agent")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(browser_router)

    _base_openapi = app.openapi

    def custom_openapi():
        schema = _base_openapi()
        schema.setdefault("paths", {})["/talk"] = _TALK_OPENAPI
        return schema

    app.openapi = custom_openapi

    @app.on_event("startup")
    async def _warm():
        ag = get_agent()
        try:
            await ag.warm()
            print("voice-agent ready", flush=True)
            if getattr(ag, "missing", []):
                print(f"voice-agent missing: {ag.missing}", flush=True)
            # device placement: LLM is a CPU sidecar (no VRAM), STT/TTS on GPU
            try:
                llm = getattr(ag, "llm", None)
                backend = getattr(getattr(llm, "config", None), "backend", "?")
                print(f"llm backend={backend} (CPU sidecar, 0 VRAM); "
                      f"stt/tts on GPU when CUDA is up", flush=True)
            except Exception:
                pass
            # report paged engine so operator sees inference engine is live
            try:
                est = _engine_stats(ag)
                if est is not None:
                    print(f"paged engine: runner={est.get('runner')} "
                          f"device={est.get('device')} "
                          f"blocks={est.get('kv_cache', {}).get('num_blocks') or est.get('kv_cache', {}).get('memory_mb') } "
                          f"prefix={est.get('prefix_cache')} "
                          f"graph={est.get('cuda_graph')}", flush=True)
                elif _paged_requested(ag):
                    print("paged engine: requested but not active (fallback to fused)", flush=True)
                else:
                    print("paged engine: off (CPU LLM sidecar owns inference)", flush=True)
            except Exception:
                pass
        except Exception as exc:  # never fail startup for import checks
            print(f"voice-agent not warmed: {exc}", flush=True)

    return app


app = create_app()


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="voice-agent API server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8003)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
