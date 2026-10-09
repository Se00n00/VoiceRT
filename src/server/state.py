"""Shared process state for the voice-term bridge routes.

Single-process singletons (agent, model profile, session bookkeeping) live
here so every route module imports the same objects.
"""

import asyncio
import base64
import time

import numpy as np

__all__ = [
    "MODEL_PROFILES",
    "_build_llm",
    "_compact_sessions",
    "_compacted_for",
    "_current_model",
    "_free_llm_gpu",
    "_lock",
    "_maybe_title",
    "_seen_sids",
    "_titled_sids",
    "_t_boot",
    "_wav_b64",
    "console_emit",
    "console_history",
    "console_subscribe",
    "console_unsubscribe",
    "get_agent",
    "note_turn_end",
    "note_turn_start",
    "qwen_leg",
]

_t_boot = time.time()
_agent = None
_switch_lock = None


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


_seen_sids: set = set()

_compacted_for: dict = {}


_titled_sids: set = set()


_turns = {"done": 0, "total_s": 0.0, "last_s": 0.0, "active": 0}


_console_events = []
_console_subs: set = set()
_CONSOLE_KEEP = 300


def console_emit(source: str, kind: str, text: str = "", sid: str = "") -> None:
    """Append a console event and fan out to live tail sockets (never raises)."""
    try:
        import time as _time

        ev = {"ts": round(_time.time(), 3), "source": str(source),
              "kind": str(kind), "text": str(text or "")[:2000],
              "sid": str(sid or "")}
        _console_events.append(ev)
        del _console_events[:-_CONSOLE_KEEP]
        for q in list(_console_subs):
            try:
                q.put_nowait(ev)
            except Exception:
                try:
                    _console_subs.discard(q)
                except Exception:
                    pass
    except Exception:
        pass


def console_history() -> list:
    return list(_console_events)


def console_subscribe(queue) -> None:
    try:
        _console_subs.add(queue)
    except Exception:
        pass


def console_unsubscribe(queue) -> None:
    try:
        _console_subs.discard(queue)
    except Exception:
        pass


def note_turn_start() -> float:
    _turns["active"] += 1
    return time.monotonic()


def note_turn_end(t0: float) -> None:
    try:
        dt = max(0.0, time.monotonic() - t0)
    except Exception:
        dt = 0.0
    _turns["active"] = max(0, _turns["active"] - 1)
    _turns["done"] += 1
    _turns["total_s"] += dt
    _turns["last_s"] = dt


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
        try:
            leg = new_llm._backend()
            if getattr(leg, "is_sidecar", False):


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


            old_leg = getattr(old, "_leg", None)
            if getattr(old_leg, "is_sidecar", False):
                old_leg.close()
        except Exception:
            pass
        try:


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
        try:
            agent.missing = [m for m in (getattr(agent, "missing", []) or [])
                             if not str(m).startswith("llm leg")]
        except Exception:
            pass


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

        from src.main import VoiceAgent, VoiceAgentConfig

        text_only = os.environ.get("VOICE_TEXT_ONLY", "").strip() == "1"
        delegate = os.environ.get("VOICE_DELEGATE", "").strip()
        cfg = VoiceAgentConfig(speak_text_turns=not text_only,
                               delegate=delegate not in ("0", "false", "no"))
        dpath = os.environ.get("VOICE_DELEGATE_CONFIG", "").strip()
        if dpath:
            cfg = cfg.model_copy(update={"delegate_config": dpath})
        _agent = VoiceAgent(cfg)
    return _agent


def _wav_b64(wav: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
    return base64.b64encode(arr.tobytes()).decode()


def qwen_leg(agent):
    """Pure-Qwen chat leg: front brain first, else the main leg when it is
    itself qwen-backed. Returns None when only a sidecar leg exists."""
    for cand in (getattr(agent, "front_llm", None), getattr(agent, "llm", None)):
        if cand is None:
            continue
        try:
            backend = str(getattr(getattr(cand, "config", None), "backend", "") or "")
        except Exception:
            continue
        if backend == "qwen":
            return cand
    return None
