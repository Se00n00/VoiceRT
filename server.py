"""voice-agent server: exactly three endpoints over one :class:`VoiceAgent`.

- ``GET /health`` — liveness, never builds the agent.
- ``GET /metrics`` — uptime, turn stats, per-node event counts.
- ``WS /talk`` — two-way socket: PCM audio in, per-node streams out.

Minimal ``/talk`` protocol (JSON text unless noted)::

    client -> server: {"type": "config", "sr": 16000,
                       "session_id": "...", "end_silence_s": 0.8}
    client -> server: {"type": "audio", "pcm_b64": "<pcm16 mono>", "sr": 16000}
    client -> server: <binary pcm16 mono frames>  (same as an audio chunk)
    client -> server: {"type": "commit"}   run a turn on the buffer now
    client -> server: {"type": "reset"}    drop the buffer + VAD state
    client -> server: {"type": "close"}    end the session

    server -> client: {"event": "ready", "session_id": ..., "sr": 16000,
                       "nodes": ["vad", "stt", "llm", "tts"]}
    server -> client: {"event": "node", "node": "vad"|"stt"|"llm"|"tts",
                       "kind": ..., "data": {...}}
                      (tts audio arrives as data.wav_b64 + sr + sentence)
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
END_SILENCE_S = 0.8
MAX_BUFFER_S = 60.0

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

    On CUDA, opts into paged LLM engine (real QwenRunner, continuous batching)
    so the inference engine is actively in the hot path for every LLM turn.
    """
    global _agent
    if _agent is None:
        from src.main import VoiceAgent, VoiceAgentConfig

        try:
            import torch
            # enable paged only when CUDA + weights are usable; warm will
            # degrade gracefully to fused path if engine fails (OOM / missing)
            if torch.cuda.is_available():
                from src.models.llm import LlmConfig

                cfg = VoiceAgentConfig(
                    llm=LlmConfig(use_paged=True, paged_blocks=32,
                                  paged_batch_size=4))
                _agent = VoiceAgent(cfg)
            else:
                _agent = VoiceAgent()
        except Exception:
            from src.main import VoiceAgent as _VA
            _agent = _VA()
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


@router.websocket("/talk")
async def talk(ws: WebSocket):
    await ws.accept()
    from engine import new_session_id

    agent = get_agent()
    sid = ws.query_params.get("session_id") or new_session_id()
    client_sr = CLIENT_SR
    end_silence_s = END_SILENCE_S
    buf = np.zeros(0, dtype=np.float32)
    speech_seen = False
    trailing_sil = 0.0
    last_vad = None
    await ws.send_json({"event": "ready", "session_id": sid, "sr": 16000,
                        "nodes": ["vad", "stt", "llm", "tts"]})

    vad = getattr(agent, "vad", None)

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
                try:
                    agent.vad.reset()
                except Exception:
                    pass
                await ws.send_json({"event": "node", "node": "vad",
                                    "kind": "reset",
                                    "data": {"buffer_s": 0.0}})
                continue
            if kind == "close":
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
                audio, buf = buf, np.zeros(0, dtype=np.float32)
                speech_seen, trailing_sil = False, 0.0
                await run_turn(audio)
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
        else:
            trailing_sil += dur
        if speech_seen and trailing_sil >= end_silence_s:
            audio, buf = buf, np.zeros(0, dtype=np.float32)
            speech_seen, trailing_sil = False, 0.0
            await run_turn(audio)


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
            '{"type":"config","sr","session_id","end_silence_s"} | '
            '{"type":"audio","pcm_b64","sr"} | binary PCM16 mono frames | '
            '{"type":"commit"} | {"type":"reset"} | {"type":"close"}.\n\n'
            "Server -> client: "
            '{"event":"ready","session_id","sr","nodes"} | '
            '{"event":"node","node":"vad"|"stt"|"llm"|"tts",'
            '"kind","data"} (tts audio arrives as data.wav_b64 + sr + '
            "sentence) | "
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

    ``router`` keeps exactly /health, /metrics, /talk (voice loop).
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
