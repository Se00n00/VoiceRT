"""Terminal bridge: FastAPI service for the Ink TUI (stock React/Ink).

Runs the local voice pipeline + LangGraph terminal harness and exposes
it over HTTP + WebSocket on :8004. ``server.py`` is untouched — this is a
separate process; run ONE of them (same 4GB GPU).

Endpoints:
- ``GET /health`` — {ok, agent_loaded, missing}
- ``POST /term/stt`` {pcm_b64, sr} — raw mic PCM -> {text}
- ``POST /term/say`` {text} — text -> {wav_b64, sr} (Kokoro)
- ``WS /term`` — full turn + confirm gate::

    client -> server: {"type": "turn", "text", "session_id", "cwd"}
    client -> server: {"type": "confirm", "ok": true|false}
    server -> client: {"event": "action", "action": {...}}
    server -> client: {"event": "observation", "observation": ...}
    server -> client: {"event": "chat", "reply": ...}
    server -> client: {"event": "audio", "wav_b64": ..., "sr": ...}
    server -> client: {"event": "confirm", "action": {...}}
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


def get_agent():
    """Process-wide VoiceAgent, built on first use."""
    global _agent
    if _agent is None:
        from src.main import VoiceAgent

        _agent = VoiceAgent()
    return _agent


def get_harness(confirm_fn=None):
    from src.agent.terminal import TerminalConfig, TerminalHarness

    agent = get_agent()
    return TerminalHarness(
        llm=agent.llm, tts=agent.tts, sessions=agent.sessions,
        config=TerminalConfig(), confirm_fn=confirm_fn)


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

    @app.on_event("startup")
    async def _warm():
        ag = get_agent()
        try:
            await ag.warm()
            print("bridge ready", flush=True)
            if getattr(ag, "missing", []):
                print(f"bridge missing: {ag.missing}", flush=True)
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
