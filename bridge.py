"""Terminal bridge: FastAPI service for the Ink TUI (stock React/Ink).

Runs the local voice pipeline + LangGraph terminal harness and exposes
it over HTTP + WebSocket on :8004. ``server.py`` is untouched — this is a
separate process; run ONE of them (same 4GB GPU).

Endpoints:
- ``GET /health`` — {ok, agent_loaded, missing}
- ``GET /model`` — {current, available:[{name, model, backend, desc, ready}]}
- ``POST /model/switch`` {name} — hot-swap the LLM leg (VRAM-safe)
- ``POST /term/stt`` {pcm_b64, sr} — raw mic PCM -> {text}
- ``POST /term/say`` {text} — text -> {wav_b64, sr} (Kokoro)
- ``WS /term`` — full turn + confirm gate::

    client -> server: {"type": "turn", "text", "session_id", "cwd"}
    client -> server: {"type": "confirm", "ok": true|false}
    server -> client: {"event": "action", "action": {...}}
    server -> client: {"event": "observation", "observation": ...}
    server -> client: {"event": "thinking", "text": ...}   (one per LLM step)
    server -> client: {"event": "token", "piece": ...}     (live stream chips)
    server -> client: {"event": "chat", "reply": ...}
    server -> client: {"event": "audio", "wav_b64": ..., "sr": ...}
    server -> client: {"event": "confirm", "action": {...}}
    server -> client: {"event": "stuck", "reason": ...}
    server -> client: {"event": "summary", "reply": ..., "session_id": ...}
    server -> client: {"event": "error", "message": ...}
"""
import asyncio
import base64
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

__all__ = ["create_app", "app"]

_t_boot = time.time()
_agent = None
_switch_lock = None  # asyncio.Lock, created lazily in loop


def _lock():
    global _switch_lock
    import asyncio as _aio

    try:
        loop = _aio.get_running_loop()
    except RuntimeError:
        loop = None
    if _switch_lock is None or (
        loop is not None and getattr(_switch_lock, "_loop", None) is not loop
    ):
        _switch_lock = _aio.Lock()
    return _switch_lock


# Model profiles the user can switch between with `/model`.
# Default is BF16 eager (plain torch, no Triton) while mixed-quant
# (4-bit bulk + BF16 important layers) is trialled. Q4_K stays as
# opt-in for the VRAM-constrained case.
MODEL_PROFILES = {
    "minicpm": {
        "name": "minicpm",
        "label": "minicpm5-1b-bf16",
        "model": "openbmb/MiniCPM5-1B",
        "backend": "minicpm",
        "desc": "default BF16 eager (no Triton, most faithful)",
        "ready": True,
    },
    "minicpm-q4k": {
        "name": "minicpm-q4k",
        "label": "minicpm5-1b-q4k",
        "model": "openbmb/MiniCPM5-1B",
        "backend": "minicpm_q4k",
        "desc": "Q4_K_M GGUF (~651MB, 4.5bpw, fastest, more hallucinations)",
        "ready": True,
    },
    "minicpm-bf16": {
        "name": "minicpm-bf16",
        "label": "minicpm5-1b-bf16",
        "model": "openbmb/MiniCPM5-1B",
        "backend": "minicpm",
        "desc": "BF16 eager alias of default",
        "ready": True,
    },
    "minicpm-q8": {
        "name": "minicpm-q8",
        "label": "minicpm5-1b-q8",
        "model": "openbmb/MiniCPM5-1B-GGUF",
        "backend": "minicpm_q4k",
        "desc": "Q8_0 (773MB, 8.5bpw, near-BF16, needs Q8 kernel — lands next)",
        "ready": False,
    },
    "qwen": {
        "name": "qwen",
        "label": "qwen3-0.6B",
        "model": "Qwen/Qwen3-0.6B",
        "backend": "qwen",
        "desc": "fused voice model fallback",
        "ready": True,
    },
}

_current_model = {"name": "minicpm"}


def _build_llm(profile: dict):
    """Construct (unwarmed) LLM leg for a profile. Monkeypatched in tests."""
    from src.models.llm import LlmConfig, LlmModel

    cfg = LlmConfig(model=profile["model"], backend=profile.get("backend", "qwen"))
    return LlmModel(cfg)


def _free_llm_gpu():
    try:
        import gc as _gc

        _gc.collect()
    except Exception:
        pass
    try:
        import torch as _t

        if _t.cuda.is_available():
            _t.cuda.empty_cache()
            try:
                _t.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


async def switch_model(name: str) -> dict:
    """Hot-swap the LLM leg. Keeps sessions/TTS/STT/VAD. Never raises."""
    name = str(name or "").strip().lower()
    profile = MODEL_PROFILES.get(name)
    if profile is None:
        return {"kind": "error",
                "message": f"unknown model {name!r}; available: {sorted(MODEL_PROFILES)}"}
    if not profile.get("ready"):
        return {"kind": "error",
                "message": f"{profile['label']} not switchable yet: {profile['desc']}"}
    if name == _current_model.get("name"):
        return {"kind": "ok", "current": name, "note": "already active"}
    lock = _lock()
    async with lock:
        if name == _current_model.get("name"):
            return {"kind": "ok", "current": name, "note": "already active"}
        agent = get_agent()
        try:
            new_llm = _build_llm(profile)
            await new_llm.warm()
        except Exception as exc:
            return {"kind": "error", "message": f"new leg failed to warm: {exc}"[:300]}
        try:  # smoke: tokenizer path alive before we commit to the swap
            await new_llm.encode([{"role": "user", "content": "ok"}])
        except Exception as exc:
            return {"kind": "error", "message": f"new leg failed smoke: {exc}"[:200]}
        old = getattr(agent, "llm", None)
        agent.llm = new_llm
        try:
            del old
        except Exception:
            pass
        _free_llm_gpu()
        try:  # drop stale llm-leg entries; keep the rest of the report
            agent.missing = [m for m in (getattr(agent, "missing", []) or [])
                             if not str(m).startswith("llm leg")]
        except Exception:
            pass
        _current_model["name"] = name
        return {"kind": "ok", "current": name, "label": profile["label"]}


def get_agent():
    """Process-wide VoiceAgent, built on first use."""
    global _agent
    if _agent is None:
        from src.main import VoiceAgent

        _agent = VoiceAgent()
    return _agent


_deep_harness = None
_deep_client = None

async def get_deep_harness():
    """Get or create the autonomous DeepAgent harness (MCP + todos)."""
    global _deep_harness, _deep_client
    if _deep_harness is not None:
        return _deep_harness
    try:
        from src.agent.deep import DeepAgentHarness

        agent_llm = get_agent().llm
        # Try to create with MCP (local python process)
        try:
            h = await DeepAgentHarness.create(agent_llm, use_mcp=True)
            _deep_harness = h
            _deep_client = getattr(h, "_mcp_client", None)
            return h
        except Exception:
            # Fallback without MCP
            h = DeepAgentHarness(agent_llm, mcp_tools=[])
            # DeepAgentHarness.__init__ already builds sync agent
            _deep_harness = h
            return h
    except Exception:
        return None

def get_harness(confirm_fn=None, use_deep: bool = False):
    """Get harness — deep if requested and available, else terminal."""
    if use_deep:
        # Sync fallback: try to get already-created deep harness
        if _deep_harness is not None:
            return _deep_harness
        # Otherwise fall back to terminal for sync callers
    from src.agent.terminal import TerminalConfig, TerminalHarness

    agent = get_agent()
    return TerminalHarness(
        llm=agent.llm, tts=agent.tts, sessions=agent.sessions,
        config=TerminalConfig(), confirm_fn=confirm_fn)

async def get_harness_async(confirm_fn=None, use_deep: bool = True):
    """Async version that can create deep harness with MCP."""
    if use_deep:
        h = await get_deep_harness()
        if h is not None:
            # Patch confirm_fn if provided (for autonomous, confirm is auto)
            if confirm_fn is not None:
                h._agent  # keep original; deep agent handles confirm via policy
            return h
    return get_harness(confirm_fn=confirm_fn, use_deep=False)


def _wav_b64(wav: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
    return base64.b64encode(arr.tobytes()).decode()


def create_app(agent=None):
    global _agent
    if agent is not None:
        _agent = agent
    app = FastAPI(title="voice-term bridge")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        try:
            loaded = bool(_agent is not None and getattr(_agent, "warmed", False))
        except Exception:
            loaded = False
        try:
            missing = list(getattr(_agent, "missing", []) or [])
        except Exception:
            missing = []
        return {"ok": True, "uptime_s": time.time() - _t_boot,
                "agent_loaded": loaded, "missing": missing}

    @app.get("/model")
    def model_info():
        cur = _current_model.get("name")
        avail = []
        for key, p in MODEL_PROFILES.items():
            avail.append({"name": key, "label": p["label"], "backend": p["backend"],
                          "desc": p["desc"], "ready": bool(p.get("ready")),
                          "current": key == cur})
        return {"current": cur, "available": avail}

    @app.post("/model/switch")
    async def model_switch(payload: dict):
        return await switch_model((payload or {}).get("name", ""))

    @app.post("/term/stt")
    async def term_stt(payload: dict):
        """Mic PCM16 mono -> transcript (VAD gate + Whisper)."""
        try:
            raw = base64.b64decode((payload or {}).get("pcm_b64", ""))
        except Exception:
            return {"kind": "error", "message": "bad pcm_b64"}
        sr = int((payload or {}).get("sr", 16000))
        wav = (np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0)
        if wav.size < 1600:
            return {"kind": "error", "message": "nothing recorded"}
        agent = get_agent()
        try:
            segs = await agent.vad.segments(wav, sr)
        except Exception as exc:
            return {"kind": "error", "message": f"VAD failed: {exc}"[:200]}
        if not segs.segments:
            return {"kind": "empty", "message": "silence"}
        try:
            res = await agent.stt.transcribe(wav, sr)
        except Exception as exc:
            return {"kind": "error", "message": f"STT failed: {exc}"[:200]}
        if not res.text.strip():
            return {"kind": "empty", "message": "STT empty"}
        return {"kind": "text", "text": res.text}

    @app.post("/term/say")
    async def term_say(payload: dict):
        """Text -> Kokoro wav (float32 b64) for Ink playback + viz."""
        text = str((payload or {}).get("text", "") or "")[:500]
        if not text.strip():
            return {"kind": "error", "message": "empty text"}
        agent = get_agent()
        try:
            out = await agent.tts.speak(text)
        except Exception as exc:
            return {"kind": "error", "message": f"TTS failed: {exc}"[:200]}
        return {"kind": "audio", "wav_b64": _wav_b64(out.wav),
                "sr": int(out.sample_rate)}

    @app.websocket("/term")
    async def term(ws: WebSocket):
        # Reader/mailbox design: ONE task owns ws.receive; the turn task
        # never blocks the socket. A naive `while receive: run_turn`
        # deadlocks on confirm (run_turn waits for a reply nobody reads).
        await ws.accept()
        inbox: asyncio.Queue = asyncio.Queue()
        pending = {"event": None, "ok": False}
        turn_task = {"task": None}
        closed = {"done": False}

        async def confirm_fn(action) -> bool:
            pending["event"] = asyncio.Event()
            pending["ok"] = False
            try:
                await ws.send_json({"event": "confirm",
                                    "action": action.as_dict()
                                    if hasattr(action, "as_dict") else dict(action)})
            except Exception:
                pending["event"] = None
                return False
            try:
                await asyncio.wait_for(pending["event"].wait(), timeout=120)
            except asyncio.TimeoutError:
                pass
            finally:
                pending["event"] = None
            return bool(pending["ok"])

        harness = get_harness(confirm_fn=confirm_fn)

        async def run_one(text, sid, cwd):
            try:
                async for event in harness.run_turn(text, session_id=sid, cwd=cwd):
                    if event.node != "term":
                        continue
                    d = dict(event.data or {})
                    wav = d.pop("wav", None)
                    if wav is not None:
                        d["wav_b64"] = _wav_b64(wav)
                    try:
                        await ws.send_json({"event": event.kind, **d})
                    except Exception:
                        break
            except Exception as exc:
                try:
                    await ws.send_json({"event": "error", "message": str(exc)[:300]})
                except Exception:
                    pass
            finally:
                turn_task["task"] = None

        async def reader():
            while not closed["done"]:
                try:
                    msg = await ws.receive_json()
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "confirm":
                    pending["ok"] = bool(msg.get("ok"))
                    if pending["event"] is not None:
                        pending["event"].set()
                    continue
                if msg.get("type") != "turn":
                    continue
                if turn_task["task"] is not None:
                    try:
                        await ws.send_json({"event": "error",
                                            "message": "turn already running"})
                    except Exception:
                        pass
                    continue
                text = str(msg.get("text", "") or "")
                if not text.strip():
                    try:
                        await ws.send_json({"event": "error",
                                            "message": "empty text"})
                    except Exception:
                        pass
                    continue
                turn_task["task"] = asyncio.create_task(
                    run_one(text, msg.get("session_id"), msg.get("cwd") or "."))

        await reader()
        closed["done"] = True
        if turn_task["task"] is not None:
            turn_task["task"].cancel()

    @app.websocket("/deep")
    async def deep(ws: WebSocket):
        """Autonomous DeepAgent (MCP + todos). Same protocol as /term but fully autonomous."""
        await ws.accept()
        # No confirm gate — deep agent is autonomous via policy
        try:
            harness = await get_deep_harness()
        except Exception as e:
            await ws.send_json({"event": "error", "message": f"deep agent unavailable: {e}"[:300]})
            return
        if harness is None:
            await ws.send_json({"event": "error", "message": "deep agent not ready"})
            return
        turn_task = {"task": None}
        closed = {"done": False}

        async def run_one(text, sid, cwd):
            try:
                async for event in harness.run_turn(text, session_id=sid, cwd=cwd):
                    d = dict(event.data or {})
                    wav = d.pop("wav", None)
                    if wav is not None:
                        d["wav_b64"] = _wav_b64(wav)
                    try:
                        await ws.send_json({"event": event.kind, **d})
                    except Exception:
                        break
            except Exception as exc:
                try:
                    await ws.send_json({"event": "error", "message": str(exc)[:300]})
                except Exception:
                    pass
            finally:
                turn_task["task"] = None

        while not closed["done"]:
            try:
                msg = await ws.receive_json()
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                continue
            if not isinstance(msg, dict) or msg.get("type") != "turn":
                continue
            if turn_task["task"] is not None:
                try:
                    await ws.send_json({"event": "error", "message": "turn already running"})
                except Exception:
                    pass
                continue
            text = str(msg.get("text", "") or "")
            if not text.strip():
                try:
                    await ws.send_json({"event": "error", "message": "empty text"})
                except Exception:
                    pass
                continue
            turn_task["task"] = asyncio.create_task(run_one(text, msg.get("session_id"), msg.get("cwd") or "."))

        closed["done"] = True
        if turn_task["task"] is not None:
            turn_task["task"].cancel()

    @app.on_event("startup")
    async def _warm():
        ag = get_agent()
        try:
            await ag.warm()
            print("bridge ready", flush=True)
            if getattr(ag, "missing", []):
                print(f"bridge missing: {ag.missing}", flush=True)
            # Pre-warm autonomous DeepAgent (MCP + todos) in background
            try:
                await get_deep_harness()
                print("deep agent ready (MCP + todos)", flush=True)
            except Exception as e:
                print(f"deep agent not ready: {e}", flush=True)
        except Exception as exc:
            print(f"bridge not warmed: {exc}", flush=True)

    return app


app = create_app()


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="voice-term bridge for Ink TUI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8004)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
