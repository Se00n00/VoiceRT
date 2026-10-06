"""Terminal bridge: FastAPI service for the Ink TUI (stock React/Ink).

Runs the unified VoiceAgent (voice + terminal turns over one deepagents
loop) and exposes it over HTTP + WebSocket on :8004. ``server.py`` stays
out of it — this is a separate process; run ONE of them (same 4GB GPU).

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
    server -> client: {"event": "title", "title": ...}  (once, after turn 1)
    server -> client: {"event": "error", "message": ...}

- ``GET /term/title?sid=`` — {title, ...} for a resumed session (no model
  call; reads the persisted name).
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
# Default is qwen (0.6B fused): same weights the two-brain front leg already
# loads, so `/model` starting there adds no download and no VRAM. Switch to
# bonsai for the 27B tool harness — that is the leg delegation escalates to,
# and it is what a bare `LlmConfig()` caller wants for agentic work.
MODEL_PROFILES = {
    "qwen": {
        "name": "qwen",
        "label": "qwen3-0.6B",
        "model": "Qwen/Qwen3-0.6B",
        "backend": "qwen",
        "desc": "0.6B fused on CUDA — DEFAULT: same weights as the two-brain front leg, chat/routing only",
        "ready": True,
    },
    "bonsai": {
        "name": "bonsai",
        "label": "bonsai-2-27b-ptq1_0",
        "model": "prism-ml/Ternary-Bonsai-2-27B",
        "backend": "bonsai",
        "desc": "27B ternary sidecar via Prism-fork llama-server (ngl auto, 16K ctx)",
        "ready": True,
    },
}

# Which LLM leg starts active. qwen (0.6B fused) matches the new LlmConfig
# default and the front leg two-brain routing already uses; /model switch
# moves to bonsai for the 27B tool harness.
_current_model = {"name": "qwen"}


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


# Sessions that ran turns in this process (compaction candidates).
_seen_sids: set = set()
# sid -> model name it was compacted for (skip repeat work per target).
_compacted_for: dict = {}
# sids that already have (or were offered) a session name. Guards the
# one-shot title call: a session is named once, on its first turn.
_titled_sids: set = set()


async def _maybe_title(agent, sid, text, ws) -> None:
    """Name the session from its opening utterance, once. Never raises.

    Fired as a background task *after* the first turn finishes rather than
    concurrently with it (opencode forks this at step 1): the front leg is
    fused on the same CUDA device as the worker here, so a concurrent call
    queues behind the real turn and taxes the user's slowest one.

    Everything about this is best-effort — a name is decoration on an exit
    card, and the TUI falls back to an offline name when it never arrives.
    """
    try:
        if not sid or sid in _titled_sids:
            return
        _titled_sids.add(sid)
        sessions = getattr(agent, "sessions", None)
        # Resumed session: a name is already on disk, so never spend a call.
        if sessions is not None:
            try:
                if sessions.get_title(sid):
                    return
            except Exception:
                pass
        from src.agent.title import generate_title

        title, _reason = await generate_title(
            getattr(agent, "front_llm", None), text)
        if not title:
            return
        if sessions is not None:
            try:
                sessions.set_title(sid, title)
            except Exception:
                pass
        await ws.send_json({"event": "title", "title": title,
                            "session_id": sid})
    except Exception:
        pass

SUMMARIZE_PROMPT = (
    "Summarize this conversation in at most 150 words: keep facts, "
    "decisions, file paths touched, and errors seen. Reply with the "
    "summary only, no preamble. Conversation:\n")


async def _compact_sessions(agent, target: str,
                            max_summary_chars: int = 2000) -> str:
    """Compact active sessions for a model switch (never raises)."""
    old_llm = getattr(agent, "llm", None)
    sessions = getattr(agent, "sessions", None)
    generate = getattr(old_llm, "generate", None)
    if not callable(generate) or sessions is None:
        return ""
    history = getattr(sessions, "history", None)
    reset = getattr(sessions, "reset", None)
    append = getattr(sessions, "append", None)
    if not all(callable(f) for f in (history, reset, append)):
        return ""
    done = 0
    for sid in sorted(_seen_sids):
        if _compacted_for.get(sid) == target:
            continue
        try:
            hist = history(sid) or []
        except Exception:
            continue
        if len(hist) <= 2:
            _compacted_for[sid] = target
            continue
        chunk = "\n".join(
            "%s: %s" % (m.get("role", "?"), m.get("content", ""))
            for m in hist[:-2])[:6000]
        try:
            res = await generate(
                [{"role": "user",
                  "content": SUMMARIZE_PROMPT + chunk}],
                max_tokens=256)
            summary = str(getattr(res, "text", "") or "").strip()
        except Exception:
            continue
        if not summary:
            continue
        try:
            reset(sid)
            append(sid, "user",
                   "[Earlier summary] " + summary[:max_summary_chars])
            for m in hist[-2:]:
                append(sid, str(m.get("role", "user") or "user"),
                       str(m.get("content", "") or ""))
        except Exception:
            continue
        _compacted_for[sid] = target
        done += 1
    if done:
        return "compacted %d session%s" % (done, "" if done == 1 else "s")
    return ""


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
        try:  # smoke: leg alive before we commit to the swap
            leg = new_llm._backend()
            if getattr(leg, "is_sidecar", False):
                # Bonsai sidecar: no local tokenizer/encode path —
                # warm() already proved /health, nothing more to smoke.
                pass
            else:
                await new_llm.encode([{"role": "user", "content": "ok"}])
        except Exception as exc:
            return {"kind": "error", "message": f"new leg failed smoke: {exc}"[:200]}
        try:
            compact_note = await _compact_sessions(agent, name)
        except Exception:
            compact_note = ""
        old = getattr(agent, "llm", None)
        try:
            # The bonsai sidecar owns a server subprocess (llama-server on
            # :8081). `del` alone would orphan it — a bonsai→qwen→bonsai
            # ping-pong would leak one server per switch. Read the already-
            # constructed leg only (never trigger a fresh _backend() build).
            old_leg = getattr(old, "_leg", None)
            if getattr(old_leg, "is_sidecar", False):
                old_leg.close()
        except Exception:
            pass
        try:
            # The new sidecar leg above only ATTACHED to the old server (its
            # port was taken); now that the old server is closed, warm again
            # so it spawns its own. Otherwise the swap commits a leg that
            # points at a dead port.
            new_leg = getattr(new_llm, "_leg", None)
            if getattr(new_leg, "is_sidecar", False) and not new_leg._health():
                await new_llm.warm()
        except Exception as exc:
            return {"kind": "error",
                    "message": f"re-warm after switch failed: {exc}"[:300]}
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
        # Explicit switch to a freshly warmed+smoked leg: mark usable and
        # clear any latched worker refusal (direct ask beats the latch).
        try:
            agent._worker_warmed = True
            agent._worker_unavailable = None
        except Exception:
            pass
        _current_model["name"] = name
        out = {"kind": "ok", "current": name, "label": profile["label"]}
        if compact_note:
            out["compacted"] = compact_note
        return out


def get_agent():
    """Process-wide VoiceAgent, built on first use.

    Two-brain delegation is ON here (and in server.py): a small front model
    talks to the user and is given no tools — it answers one yes/no question
    per turn, and a YES hands the raw user text to the worker leg with the
    full harness. Off with ``VOICE_DELEGATE=0`` for the old single-brain
    behaviour.

    VOICE_TEXT_ONLY=1 skips the TTS leg (warm + per-turn speak): text
    turns stay VRAM-clean on 4GB cards next to a 27B LLM. Voice turns
    and /term/say still fail loudly if called — this flag declares a
    text-only backend, it doesn't remove the voice paths.
    """
    global _agent
    if _agent is None:
        import os

        from dataclasses import replace

        from src.main import VoiceAgent, VoiceAgentConfig

        text_only = os.environ.get("VOICE_TEXT_ONLY", "").strip() == "1"
        delegate = os.environ.get("VOICE_DELEGATE", "").strip()
        cfg = VoiceAgentConfig(speak_text_turns=not text_only,
                               delegate=delegate not in ("0", "false", "no"))
        dpath = os.environ.get("VOICE_DELEGATE_CONFIG", "").strip()
        if dpath:
            cfg = replace(cfg, delegate_config=dpath)
        _agent = VoiceAgent(cfg)
    return _agent


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

    @app.get("/term/title")
    def term_title(sid: str = ""):
        """Persisted name for a session, so a resume keeps it.

        Read-only and model-free on purpose: the TUI calls this once on
        mount, and ``voicert -s <id>`` followed by an immediate quit never
        runs a turn, so there is nothing to generate a title from.
        """
        sid = str(sid or "").strip()
        if not sid:
            return {"title": "", "session_id": ""}
        try:
            title = get_agent().sessions.get_title(sid)
        except Exception:
            title = ""
        return {"title": title or "", "session_id": sid}

    @app.get("/term/history")
    def term_history(sid: str = ""):
        """Full persisted transcript for a session (past conversations).

        Reads ``sessions/<sid>.json`` straight off disk — no model, no
        TTL prune — so the UI can restore user/assistant turns plus the
        richer ``thinking`` / ``tool`` / ``cot`` roles the agent never
        feeds back into the LLM window (see ``_lc_messages``: user and
        assistant only). Touches the file so the session sweeper does
        not reap it while it is being viewed.
        """
        import json as _json
        import os as _os

        from src.agent.memory import _safe_sid

        sid = str(sid or "").strip()
        if not sid:
            return {"session_id": "", "title": "", "messages": []}
        try:
            base = getattr(get_agent().sessions, "sessions_dir", "sessions") or "sessions"
            path = _os.path.join(str(base), _safe_sid(sid) + ".json")
            with open(path, "r", encoding="utf-8") as f:
                payload = _json.load(f)
            try:
                _os.utime(path, None)
            except Exception:
                pass
            msgs = payload.get("messages", []) or []
            msgs = [
                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                for m in msgs
                if isinstance(m, dict) and str(m.get("content", "") or "").strip()
            ]
            return {
                "session_id": str(payload.get("session_id", sid) or sid),
                "title": str(payload.get("title", "") or ""),
                "messages": msgs,
            }
        except Exception:
            return {"session_id": sid, "title": "", "messages": []}

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

        harness = get_agent()

        async def run_one(text, sid, cwd):
            try:
                async for event in harness.run_text(text, session_id=sid, cwd=cwd,
                                                    confirm_fn=confirm_fn):
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
            except Exception as exc:  # noqa: BLE001 - surface, never crash
                try:
                    await ws.send_json({"event": "error", "message": str(exc)[:300]})
                except Exception:
                    pass
            finally:
                turn_task["task"] = None
                # Name the session once the turn is over, so the title call
                # never competes with it for the GPU.
                if sid and not closed["done"]:
                    try:
                        asyncio.create_task(
                            _maybe_title(harness, str(sid), text, ws))
                    except RuntimeError:
                        pass  # loop already closing

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
                if msg.get("session_id"):
                    _seen_sids.add(str(msg.get("session_id")))
                turn_task["task"] = asyncio.create_task(
                    run_one(text, msg.get("session_id"), msg.get("cwd") or "."))

        await reader()
        closed["done"] = True
        if turn_task["task"] is not None:
            turn_task["task"].cancel()

    @app.websocket("/deep")
    async def deep(ws: WebSocket):
        """Autonomous agent over the same VoiceAgent (no confirm gate)."""
        await ws.accept()
        harness = get_agent()
        turn_task = {"task": None}
        closed = {"done": False}

        async def run_one(text, sid, cwd):
            try:
                # autonomous: policy still denies breakers, confirms auto-pass
                async for event in harness.run_text(text, session_id=sid, cwd=cwd,
                                                    confirm_fn=lambda action: True):
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
            if msg.get("session_id"):
                _seen_sids.add(str(msg.get("session_id")))
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
