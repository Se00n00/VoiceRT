"""Shared process state for the voice-term bridge routes.

Single-process singletons (agent, model profile, session bookkeeping) live
here so every route module imports the same objects.
"""

import asyncio
import base64
import time

import numpy as np

from src.agent.store import VoicertStore
from src.setup import VoicertSetup

__all__ = [
    "MODEL_PROFILES",
    "_brain",
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
    "get_agent",
    "get_brain_mode",
    "note_turn_end",
    "note_turn_start",
    "qwen_leg",
    "switch_brain_mode",
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
        "desc": "0.6B fused on CUDA — DEFAULT local leg, chat/routing only",
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
    "groq": {
        "name": "groq",
        "label": "gpt-oss-120b",
        "model": "openai/gpt-oss-120b",
        "backend": "groq",
        "desc": "Groq cloud via LangChain (no GPU; needs GROQ_API_KEY)",
        "ready": True,
    },
    "anthropic": {
        "name": "anthropic",
        "label": "claude-sonnet",
        "model": "claude-sonnet-4-5",
        "backend": "anthropic",
        "desc": "Claude via LangChain (no GPU; needs ANTHROPIC_API_KEY)",
        "ready": True,
    },
    "openai": {
        "name": "openai",
        "label": "gpt-4o-mini",
        "model": "gpt-4o-mini",
        "backend": "openai",
        "desc": "OpenAI via LangChain (no GPU; needs OPENAI_API_KEY)",
        "ready": True,
    },
    "openrouter": {
        "name": "openrouter",
        "label": "openrouter",
        "model": "deepseek/deepseek-chat",
        "backend": "openrouter",
        "desc": "Any OpenRouter model via LangChain (needs OPENROUTER_API_KEY)",
        "ready": True,
    },
    "deepseek": {
        "name": "deepseek",
        "label": "deepseek-chat",
        "model": "deepseek-chat",
        "backend": "deepseek",
        "desc": "DeepSeek via LangChain (no GPU; needs DEEPSEEK_API_KEY)",
        "ready": True,
    },
    "compat": {
        "name": "compat",
        "label": "openai-compat",
        "model": "",
        "backend": "openai_compat",
        "desc": "Any OpenAI-compatible endpoint (needs OPENAI_COMPAT_BASE_URL + OPENAI_COMPAT_MODEL)",
        "ready": True,
    },
}


DEFAULT_MODEL = "groq"
"""Boot profile (groq/oss-120b); falls back to qwen without a key."""

_current_model = {"name": DEFAULT_MODEL}


# Brain mode: "voice" runs the full voice agent (voice pipeline on top of
# the main LLM harness); "harness" runs just the LLM harness (text turns
# only, no duplex voice). The LLM harness itself (cloud/local via
# MODEL_PROFILES) is shared by both brains. The harness brain defaults to
# groq: entering it stashes the voice brain's model in "prev_model" and
# auto-switches, leaving it restores the stash.
_brain = {"mode": "voice", "prev_model": ""}


def get_brain_mode() -> str:
    """Current brain mode, "voice" or "harness".

    ..
    """
    mode = str((_brain or {}).get("mode") or "voice").lower()
    return mode if mode in ("voice", "harness") else "voice"


async def switch_brain_mode(mode: str) -> dict:
    """Switch brains, defaulting the harness to groq. Never raises.

    Entering "harness" auto-switches the shared harness to groq (the
    local qwen leg is a chat/router, not an agentic brain) and stashes
    the voice brain's model; returning to "voice" restores the stash
    when it is still groq. Model failures ride along as out["model"],
    the brain switch itself still lands.
    ..
    """
    name = str(mode or "").strip().lower()
    if name not in ("voice", "harness"):
        return {"kind": "error",
                "message": f"unknown brain {mode!r}; available: ['voice', 'harness']"}
    if name == get_brain_mode():
        return {"kind": "ok", "mode": name, "note": "already active"}
    _brain["mode"] = name
    out: dict = {"kind": "ok", "mode": name}
    try:
        if name == "harness":
            if (_current_model.get("name") or "") != "groq":
                _brain["prev_model"] = str(_current_model.get("name") or "")
                out["model"] = await switch_model("groq")
        else:
            prev = str(_brain.get("prev_model") or "")
            _brain["prev_model"] = ""
            if prev and prev in MODEL_PROFILES and (_current_model.get("name") or "") == "groq":
                out["model"] = await switch_model(prev)
    except Exception as exc:
        out["model"] = {"kind": "error", "message": str(exc)[:200]}
    return out


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
            getattr(agent, "llm", None), text)
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
    if sessions is None:
        return ""
    generate = getattr(old_llm, "generate", None) if old_llm is not None else None
    chat_model = getattr(agent, "chat_model", None)
    if not callable(generate) and chat_model is None:
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
            if callable(generate):
                res = await generate(
                    [{"role": "user",
                      "content": SUMMARIZE_PROMPT + chunk}],
                    max_tokens=256)
                summary = str(getattr(res, "text", "") or "").strip()
            else:
                from langchain_core.messages import HumanMessage

                from src.models.provider import ChatModel

                out = await chat_model.ainvoke(
                    [HumanMessage(content=SUMMARIZE_PROMPT + chunk)])
                summary = ChatModel.chat_text(getattr(out, "content", "")).strip()
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


async def _switch_to_cloud(name: str, profile: dict, agent) -> dict:
    """Swap the worker to a stock cloud chat model. Returns the switch dict.

    ..
    """
    from src.models.provider import ChatModel

    backend = str(profile.get("backend", "") or "").lower()
    try:
        chat_model = ChatModel.get(backend, str(profile.get("model", "") or ""))
    except Exception as exc:
        return {"kind": "error",
                "message": f"new leg failed to warm: {exc}"[:300]}
    try:
        from langchain_core.messages import HumanMessage

        await asyncio.wait_for(chat_model.ainvoke(
            [HumanMessage(content="ok")]), timeout=30)
    except Exception as exc:
        return {"kind": "error",
                "message": f"new leg failed smoke: {exc}"[:200]}
    try:
        compact_note = await _compact_sessions(agent, name)
    except Exception:
        compact_note = ""
    old = getattr(agent, "llm", None)
    try:
        old_leg = getattr(old, "_leg", None) if old is not None else None
        if getattr(old_leg, "is_sidecar", False):
            old_leg.close()
    except Exception:
        pass
    agent.llm = None
    try:
        agent.active_model_label = str(profile.get("label", "") or name)
        agent.chat_model = chat_model
        agent.agent = agent.build_longrunning_agent()
    except Exception:
        pass
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
        agent.worker.warmed = True
        agent.worker.unavailable = None
    except Exception:
        pass
    _current_model["name"] = name
    out = {"kind": "ok", "current": name, "label": profile["label"]}
    if compact_note:
        out["compacted"] = compact_note
    return out


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
            from src.models.provider import ChatModel

            if ChatModel.is_cloud_backend(str(profile.get("backend", "") or "")):
                return await _switch_to_cloud(name, profile, agent)
        except Exception:
            pass
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
            old_leg = getattr(old, "_leg", None) if old is not None else None
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
            from src.models.chat_model import LocalChatModel

            agent.active_model_label = str(profile.get("label", "") or name)
            agent.chat_model = LocalChatModel(llm=new_llm)
            agent.agent = agent.build_longrunning_agent()
        except Exception:
            pass
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
            agent.worker.warmed = True
            agent.worker.unavailable = None
        except Exception:
            pass
        _current_model["name"] = name
        out = {"kind": "ok", "current": name, "label": profile["label"]}
        if compact_note:
            out["compacted"] = compact_note
        return out


def _apply_default_profile(agent) -> None:
    """Swap a fresh agent to DEFAULT_MODEL without a smoke call.

    Cloud defaults pre-wire when a key resolves (first turn surfaces
    errors); without one the qwen build stands and the label follows.
    A failed graph rebuild restores the previous legs. Never raises.
    ..
    """
    try:
        name = str(_current_model.get("name", "") or "").lower()
        profile = MODEL_PROFILES.get(name) or {}
        if name == "qwen" or not profile.get("ready"):
            return
        from src.models.provider import ChatModel

        backend = str(profile.get("backend", "") or "")
        if not ChatModel.is_cloud_backend(backend):
            return
        old_llm, old_chat = (getattr(agent, "llm", None),
                             getattr(agent, "chat_model", None))
        try:
            agent.chat_model = ChatModel.get(
                backend, str(profile.get("model", "") or ""))
        except SystemExit:
            _current_model["name"] = "qwen"
            return
        except Exception:
            return
        agent.llm = None
        try:
            agent.agent = agent.build_longrunning_agent()
        except Exception:
            agent.llm, agent.chat_model = old_llm, old_chat
            _current_model["name"] = "qwen"
            return
        agent.worker.warmed = True
    except Exception:
        pass


def get_agent():
    """Process-wide VoiceAgent, built on first use.

    Every turn runs the full worker graph with tools and memory.

    VOICE_TEXT_ONLY=1 skips the TTS leg (warm + per-turn speak): text
    turns stay VRAM-clean on 4GB cards next to a 27B LLM. Voice turns
    and /term/say still fail loudly if called — this flag declares a
    text-only backend, it doesn't remove the voice paths.
    """
    global _agent
    if _agent is None:
        import os

        from src.agent.long_running import LongRunningConfig
        from src.agent.voice_agent.agent import VoiceAgent, VoiceAgentConfig

        try:
            VoicertSetup.ensure()
            VoicertStore().log("info", "backend start")
        except Exception:
            pass
        text_only = os.environ.get("VOICE_TEXT_ONLY", "").strip() == "1"
        harness = LongRunningConfig(speak_text_turns=not text_only)
        cfg = VoiceAgentConfig(harness=harness)
        _agent = VoiceAgent(cfg)
        _apply_default_profile(_agent)
        try:
            _agent.active_model_label = str(
                MODEL_PROFILES.get(
                    str(_current_model.get("name", "") or ""),
                    {}).get("label", "") or "gpt-oss-120b")
        except Exception:
            pass
    return _agent


def _wav_b64(wav: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
    return base64.b64encode(arr.tobytes()).decode()


def qwen_leg(agent):
    """Pure-Qwen chat leg: the main leg when it is qwen-backed.

    Returns None when only a sidecar leg exists."""
    for cand in (getattr(agent, "llm", None),):
        if cand is None:
            continue
        try:
            backend = str(getattr(getattr(cand, "config", None), "backend", "") or "")
        except Exception:
            continue
        if backend == "qwen":
            return cand
    return None
