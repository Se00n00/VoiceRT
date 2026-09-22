"""Shared runner for agent-level evals (GAIA L1, TerminalBench-style).

One warmed VoiceAgent (LLM leg only — no VAD/STT/TTS weights) answers each
task via run_text in a fresh temp workdir with auto-approving confirm
(policy denies still apply — that's part of what's measured).

Timeouts bound every task; everything is recorded for grading.
"""
import asyncio
import os
import tempfile
import time

CONFIRM_LOG = "auto-approve (deny verdicts still block)"


async def warm_agent(model="openbmb/MiniCPM5-1B", backend="minicpm",
                     sessions_dir=None):
    """Build VoiceAgent with warmed LLM leg, TTS disabled."""
    from src.main import VoiceAgent, VoiceAgentConfig
    from src.models.llm import LlmConfig, LlmModel, assert_cuda_leg

    cfg = VoiceAgentConfig(
        sessions_dir=sessions_dir or tempfile.mkdtemp(prefix="eval-sess-"))
    agent = VoiceAgent(cfg)
    agent.llm = LlmModel(LlmConfig(model=model, backend=backend))
    print(f"warming LLM {model} [{backend}] ...", flush=True)
    await agent.llm.warm()
    # the agent loop runs through chat_model: rewrap so it sees this leg
    from src.agent.chat_model import LocalChatModel

    agent.chat_model = LocalChatModel(llm=agent.llm)
    agent.tts = None
    print(f"agent ready (llm-only) on {assert_cuda_leg(agent.llm)}.", flush=True)
    return agent


def make_workdir(files=None):
    """Fresh temp cwd with optional seed files {relpath: content}."""
    d = tempfile.mkdtemp(prefix="eval-task-")
    for rel, content in (files or {}).items():
        full = os.path.join(d, rel)
        os.makedirs(os.path.dirname(full) or d, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    return d


async def run_task(agent, prompt, workdir, timeout_s=600, session_id=None):
    """Run one task; returns dict with reply, steps, events, seconds."""
    import uuid

    sid = session_id or f"eval-{uuid.uuid4().hex[:8]}"
    confirmed = {"n": 0}

    def confirm_fn(action):
        confirmed["n"] += 1
        return True

    t0 = time.perf_counter()
    events = []
    try:
        async def _collect():
            out = []
            async for ev in agent.run_text(prompt, session_id=sid,
                                           cwd=workdir, confirm_fn=confirm_fn):
                out.append({"node": ev.node, "kind": ev.kind,
                            "data": _jsonable(ev.data or {})})
            return out

        events = await asyncio.wait_for(_collect(), timeout=timeout_s)
        timed_out = False
    except asyncio.TimeoutError:
        timed_out = True
    dt = time.perf_counter() - t0
    replies = [e["data"].get("reply", "") for e in events
               if e["kind"] == "chat" and e["data"].get("reply")]
    summaries = [e for e in events if e["kind"] == "summary"]
    actions = [e for e in events if e["kind"] == "action"]
    return {
        "reply": replies[-1] if replies else "",
        "replies": replies,
        "summary": (summaries[-1]["data"] if summaries else {}),
        "steps": len(actions),
        "actions": [(a["data"].get("action", {}) or {}).get("action", "?")
                     for a in actions],
        "confirms": confirmed["n"],
        "timed_out": timed_out,
        "seconds": round(dt, 2),
        "n_events": len(events),
    }


def _jsonable(obj, depth=0):
    """Best-effort JSON-safe projection of event data (caps size)."""
    import numpy as _np

    if depth > 3:
        return "…"
    if isinstance(obj, dict):
        return {str(k)[:64]: _jsonable(v, depth + 1)
                for k, v in list(obj.items())[:24]}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in list(obj)[:24]]
    if isinstance(obj, _np.ndarray):
        return f"<ndarray {obj.shape} {obj.dtype}>"
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    try:
        s = str(obj)
    except Exception:
        return "…"
    return s if len(s) <= 2000 else s[:2000] + "…[truncated]"
