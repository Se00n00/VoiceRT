"""HTTP routes for the voice pipeline.

Endpoints:
- GET  /health   (never loads the engine: liveness-safe)
- GET  /metrics  (per-endpoint counts/errors/latency, uptime, vram)
- POST /v1/vad         wav upload -> {segments, audio_dur_s, n_segments}
- POST /v1/transcribe  wav upload -> {text, rtf, ttfs, dur}
- POST /v1/chat        {prompt, max_tokens, stream} -> JSON or SSE tokens
- POST /v1/speak       {text} -> audio/wav bytes
- POST /v1/voice       wav upload -> full turn {text, reply, wav_b64,
                       ttfa_s, total_s, vram_mb}

Production guards (all measured, see README):
- audio capped at MAX_AUDIO_S (whisper context is 30s; longer uploads
  would only burn VRAM/time) -> 413
- empty audio -> 400
- engine semaphore MAX_INFLIGHT=4 (measured serialization point: c=4
  halves throughput) -> 503, never unbounded queueing
- missing legs (weights absent) -> 503, not 500
- request schemas bound prompt/tokens/text sizes (pydantic -> 422)
"""
import asyncio
import base64
import io
import os
import tempfile
import threading
import time

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from server.schemas import ChatReq, SpeakReq

router = APIRouter()

MAX_AUDIO_S = 60.0
MAX_INFLIGHT = 4

_engine = None
_t_boot = time.time()
_metrics = {}  # endpoint -> [count, errors, total_s]
_metrics_lock = threading.Lock()

# -- admission control: FIFO queue + VRAM-derived capacity ----------------
# _sched serializes turns (single worker: c=4 halves throughput, measured).
# _capacity is the startup plan from runtime.capacity (see configure_capacity);
# pre-startup the conservative defaults below apply so imports/tests work.
from runtime.scheduler import FIFOScheduler

_sched = FIFOScheduler(MAX_INFLIGHT)
_QUEUE_TIMEOUT_S = 10.0
_capacity = {"max_sessions": 1000, "max_inflight": MAX_INFLIGHT,
             "generation_length": 48, "queue_timeout_s": _QUEUE_TIMEOUT_S,
             "conservative": True}


def configure_capacity(plan):
    """Apply a startup capacity plan (called once from app startup).

    Resizes the FIFO scheduler, the queue timeout, the session-store cap,
    and publishes the plan on /health. Safe to call twice (last wins).
    """
    global _QUEUE_TIMEOUT_S
    _capacity.clear()
    _capacity.update(dict(plan))
    try:
        _sched.max_concurrency = max(1, int(plan.get("max_inflight",
                                                     MAX_INFLIGHT)))
    except Exception:
        pass
    try:
        _QUEUE_TIMEOUT_S = float(plan.get("queue_timeout_s",
                                          _QUEUE_TIMEOUT_S))
    except Exception:
        pass
    try:
        if engine_loaded():
            get_engine().sessions.max_sessions = max(
                1, int(plan.get("max_sessions", 1000)))
    except Exception:
        pass
    return dict(_capacity)


def get_capacity():
    """Current serving plan (conservative defaults before startup)."""
    return dict(_capacity)


def get_engine():
    """Lazily construct the singleton VoiceEngine."""
    global _engine
    if _engine is None:
        from engine.engine import VoiceEngine

        _engine = VoiceEngine()
    return _engine


def engine_loaded():
    return _engine is not None


def _record(endpoint, dt, err=False):
    with _metrics_lock:
        c, e, t = _metrics.get(endpoint, (0, 0, 0.0))
        _metrics[endpoint] = (c + 1, e + int(err), t + dt)


def _acquire_or_503():
    """Take a FIFO queue ticket, waiting up to the configured timeout.

    Burst traffic queues fairly instead of thundering-herd; only a full
    queue (timeout) becomes 503 + Retry-After. Queue wait is recorded for
    /metrics. Returns the ticket; every ticket must reach _release().
    """
    t0 = time.perf_counter()
    try:
        ticket = _sched.acquire(blocking=True, timeout=_QUEUE_TIMEOUT_S)
    except Exception:
        ticket = None
    if ticket is None:
        _record("queue_timeout", time.perf_counter() - t0, err=True)
        raise HTTPException(
            status_code=503,
            detail=(f"server saturated ({_sched.max_concurrency} in flight, "
                    f"queue waited {_QUEUE_TIMEOUT_S:.0f}s); retry"),
            headers={"Retry-After": "2"},
        )
    _record("queue_wait", time.perf_counter() - t0)
    return ticket


def _release(ticket):
    try:
        _sched.release(ticket)
    except Exception:
        pass


def _leg_error(exc):
    from runtime.memory import MemoryBudgetExceeded

    msg = str(exc)
    if isinstance(exc, MemoryBudgetExceeded):
        raise HTTPException(status_code=503, detail=msg,
                            headers={"Retry-After": "2"})
    if "leg(s) not loaded" in msg or "not loaded" in msg:
        raise HTTPException(status_code=503, detail=msg)
    raise HTTPException(status_code=500, detail="engine error")


def _load_wav_sync(data: bytes):
    """Raw wav bytes -> (mono float audio, sample rate) via soundfile."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        audio, sr = sf.read(path)
    finally:
        os.unlink(path)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return audio, sr


async def _parse_upload(data: bytes):
    """Upload bytes -> (audio, sr); malformed uploads are 400, not 500."""
    try:
        return await asyncio.to_thread(_load_wav_sync, data)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="unreadable audio upload")


def _guard_wav(audio, sr):
    dur = len(audio) / float(sr)
    if not len(audio):
        raise HTTPException(status_code=400, detail="empty audio")
    if dur > MAX_AUDIO_S:
        raise HTTPException(
            status_code=413,
            detail=f"audio {dur:.1f}s exceeds {MAX_AUDIO_S:.0f}s cap",
        )
    return dur


def _vram_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return None


@router.get("/health")
def health():
    out = {
        "ok": True,
        "uptime_s": time.time() - _t_boot,
        "engine_loaded": engine_loaded(),
        "legs": ["vad", "transcribe", "chat", "speak", "voice"],
        "capacity": get_capacity(),
    }
    if engine_loaded():
        try:
            eng = get_engine()
            missing = getattr(eng, "missing", [])
            out["missing"] = list(missing)
            out["vram_mb"] = _vram_mb()
            try:
                from models.qwen import HAVE_TRITON_KERNELS as q
                from models.tts import HAVE_TRITON_KERNELS as t
                from models.whisper import HAVE_TRITON_KERNELS as w
                from models.silero_vad.kernels import HAVE_TRITON_LSTM as v

                out["triton"] = {
                    "qwen": bool(q),
                    "whisper": bool(w),
                    "tts": bool(t),
                    "vad_lstm": bool(v),
                }
            except Exception:
                pass
        except Exception as exc:
            out["ok"] = False
            out["error"] = str(exc)[:200]
    return out


@router.get("/metrics")
def metrics():
    with _metrics_lock:
        eps = {
            k: {
                "count": c,
                "errors": e,
                "mean_s": (t / c) if c else 0.0,
            }
            for k, (c, e, t) in _metrics.items()
        }
    return {
        "uptime_s": time.time() - _t_boot,
        "inflight": _sched.running,
        "queue": _sched.stats(),
        "queue_wait_s": _mean("queue_wait"),
        "vram_mb": _vram_mb(),
        "endpoints": eps,
    }


def _mean(endpoint):
    with _metrics_lock:
        c, _, t = _metrics.get(endpoint, (0, 0, 0.0))
    return (t / c) if c else 0.0


@router.post("/v1/vad")
async def vad(f: UploadFile):
    t0 = time.perf_counter()
    try:
        data = await f.read()
        audio, sr = await _parse_upload(data)
        _guard_wav(audio, sr)
        ticket = _acquire_or_503()
        try:
            eng = get_engine()
            segs = await asyncio.to_thread(eng.vad_segments, audio)
        finally:
            _release(ticket)
        segs = [[float(a), float(b)] for a, b in segs["segments"]]
        _record("vad", time.perf_counter() - t0)
        return {
            "segments": segs,
            "audio_dur_s": len(audio) / float(sr),
            "n_segments": len(segs),
        }
    except HTTPException:
        _record("vad", time.perf_counter() - t0, err=True)
        raise
    except RuntimeError as exc:
        _record("vad", time.perf_counter() - t0, err=True)
        _leg_error(exc)
    except Exception:
        _record("vad", time.perf_counter() - t0, err=True)
        raise HTTPException(status_code=500, detail="engine error")


@router.post("/v1/transcribe")
async def transcribe(f: UploadFile):
    t0 = time.perf_counter()
    try:
        data = await f.read()
        audio, sr = await _parse_upload(data)
        _guard_wav(audio, sr)
        ticket = _acquire_or_503()
        try:
            eng = get_engine()
            r = await asyncio.to_thread(eng.transcribe, audio, sr)
        finally:
            _release(ticket)
        _record("transcribe", time.perf_counter() - t0)
        return r
    except HTTPException:
        _record("transcribe", time.perf_counter() - t0, err=True)
        raise
    except RuntimeError as exc:
        _record("transcribe", time.perf_counter() - t0, err=True)
        _leg_error(exc)
    except Exception:
        _record("transcribe", time.perf_counter() - t0, err=True)
        raise HTTPException(status_code=500, detail="engine error")


@router.post("/v1/chat")
def chat(req: ChatReq):
    from engine.session import new_session_id

    t0 = time.perf_counter()
    sid = req.session_id or new_session_id()
    ticket = _acquire_or_503()
    try:
        eng = get_engine()
    except Exception:
        _release(ticket)
        raise
    try:
        if not req.stream:
            try:
                r = eng.chat(req.prompt, max_tokens=req.max_tokens,
                             stream=False, session_id=sid, reset=req.reset)
            finally:
                _release(ticket)
            text = r.get("text", "")
            ttft = r.get("ttft", r.get("ttft_s", 0.0))
            tps = r.get("tps", r.get("decode_tps", 0.0))
            _record("chat", time.perf_counter() - t0)
            return {"text": text, "ttft_s": ttft, "tps": tps,
                    "session_id": sid}

        def sse():
            from engine.streaming import done_frame, format_sse

            try:
                if req.reset:
                    eng.sessions.reset(sid)
                ids = eng.prompt_ids(req.prompt, sid)
                out_ids = []
                for tok_id, _ in eng.llm.generate_stream(ids, req.max_tokens):
                    out_ids.append(tok_id)
                    yield format_sse(None, repr(eng.tok.decode([tok_id])))
                eng.remember(sid, req.prompt,
                             eng.tok.decode(out_ids, skip_special_tokens=True))
                yield done_frame()
            finally:
                _record("chat_stream", time.perf_counter() - t0)
                _release(ticket)

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"X-Session-Id": sid})
    except HTTPException:
        _record("chat", time.perf_counter() - t0, err=True)
        raise
    except RuntimeError as exc:
        _record("chat", time.perf_counter() - t0, err=True)
        _leg_error(exc)
    except Exception:
        _record("chat", time.perf_counter() - t0, err=True)
        raise HTTPException(status_code=500, detail="engine error")


@router.post("/v1/speak")
def speak(req: SpeakReq):
    import soundfile as sf

    t0 = time.perf_counter()
    ticket = _acquire_or_503()
    try:
        eng = get_engine()
        r = eng.speak(req.text)
        buf = io.BytesIO()
        sf.write(buf, r["wav"], r["sr"], format="WAV")
        _record("speak", time.perf_counter() - t0)
        return Response(
            content=buf.getvalue(),
            media_type="audio/wav",
            headers={"X-Synth-S": str(r.get("synth_s", 0.0))},
        )
    except HTTPException:
        _record("speak", time.perf_counter() - t0, err=True)
        raise
    except RuntimeError as exc:
        _record("speak", time.perf_counter() - t0, err=True)
        _leg_error(exc)
    except Exception:
        _record("speak", time.perf_counter() - t0, err=True)
        raise HTTPException(status_code=500, detail="engine error")
    finally:
        _release(ticket)


@router.post("/v1/voice")
async def voice(f: UploadFile, session_id: str = Form("")):
    import numpy as np

    from engine.session import new_session_id

    t0 = time.perf_counter()
    sid = session_id or new_session_id()
    try:
        data = await f.read()
        audio, sr = await _parse_upload(data)
        _guard_wav(audio, sr)
        ticket = _acquire_or_503()
        try:
            eng = get_engine()
            r = await asyncio.to_thread(eng.stream_turn, audio, sr, sid)
        finally:
            _release(ticket)
        wav = np.asarray(r["wav"], dtype=np.float32)
        wav_b64 = base64.b64encode(wav.tobytes()).decode()
        _record("voice", time.perf_counter() - t0)
        return {
            "text": r["text"],
            "reply": r["reply"],
            "wav_b64": wav_b64,
            "ttfa_s": r["ttfa_s"],
            "total_s": r["total_s"],
            "vram_mb": r.get("vram_mb"),
            "session_id": sid,
        }
    except HTTPException:
        _record("voice", time.perf_counter() - t0, err=True)
        raise
    except RuntimeError as exc:
        _record("voice", time.perf_counter() - t0, err=True)
        _leg_error(exc)
    except Exception:
        _record("voice", time.perf_counter() - t0, err=True)
        raise HTTPException(status_code=500, detail="engine error")


@router.get("/v1/sessions")
def sessions():
    """Sessions held vs the VRAM-derived serving plan."""
    try:
        stats = get_engine().sessions.stats()
    except Exception:
        stats = {"sessions": 0, "turns": 0}
    stats["max_sessions"] = _capacity.get("max_sessions", 1000)
    stats["generation_length"] = _capacity.get("generation_length", 48)
    return stats


@router.delete("/v1/session/{sid}")
def clear_session(sid: str):
    """Drop one session's history. Frontend calls this on chat reset."""
    try:
        dropped = get_engine().sessions.drop(sid)
    except Exception:
        dropped = False
    return {"ok": True, "cleared": bool(dropped), "session_id": sid}
