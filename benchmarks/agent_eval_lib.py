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


async def warm_agent(model="Qwen/Qwen3-0.6B", backend="qwen",
                     sessions_dir=None, max_tokens=None):
    """Build VoiceAgent with warmed LLM leg, TTS disabled."""
    from src.main import VoiceAgent, VoiceAgentConfig
    from src.models.llm import LlmConfig, LlmModel, assert_ready_leg

    # sessions_dir kept for API compat (JsonSessionMemory persists per sid).
    cfg = VoiceAgentConfig(
        **({"sessions_dir": sessions_dir} if sessions_dir else {}))
    agent = VoiceAgent(cfg)
    leg_kw = {} if max_tokens is None else {"max_tokens": int(max_tokens)}
    if backend in ("groq", "gemini") and model != "Qwen/Qwen3-0.6B":
        # Cloud legs ignore `model` (qwen HF id): --model names their model.
        key = "groq_model" if backend == "groq" else "gemini_model"
        agent.llm = LlmModel(LlmConfig(backend=backend, **{key: model},
                                       **leg_kw))
    else:
        agent.llm = LlmModel(LlmConfig(model=model, backend=backend,
                                       **leg_kw))
    print(f"warming LLM {model} [{backend}] ...", flush=True)
    await agent.llm.warm()
    # the agent loop runs through chat_model: rewrap so it sees this leg
    # (VoiceAgent rebuilds the graph every turn, so the rewrap takes effect).
    # Keep the MCP-fs mapping flag: without it terminal read/write/edit die
    # on the tool-name collision when extras are attached.
    from src.agent.chat_model import LocalChatModel

    agent.chat_model = LocalChatModel(
        llm=agent.llm, use_mcp_fs=bool(agent._extra_tools))
    agent.tts = None
    print(f"agent ready (llm-only) on {assert_ready_leg(agent.llm)}.", flush=True)
    return agent


def anchor_workdir(agent, workdir):
    """Point one task's file ops at its temp workdir.

    Two channels, both rooted per task (the graph rebuilds every turn,
    so swaps take effect on the next turn):
    - MCP servers snapshot os.environ at spawn, so re-spawn the extras
      after setting VOICE_WORKDIR (seconds per task; the LLM stays warm).
    - the shell backend is rooted at construction: swap in a fresh
      LocalShellBackend rooted at the workdir (cheap object, no spawn).
    """
    import os

    from src.agent.mcp.client import load_extra_tools

    os.environ["VOICE_WORKDIR"] = workdir
    try:
        from deepagents.backends import LocalShellBackend

        agent.backend = LocalShellBackend(root_dir=workdir, timeout=30)
    except Exception:
        pass
    try:
        agent._mcp_client, agent._extra_tools = load_extra_tools()
    except Exception:
        pass
    agent.chat_model.use_mcp_fs = bool(agent._extra_tools)


def make_workdir(files=None, attachments=None, src_root="."):
    """Fresh temp cwd with seed files {relpath: content} + attachments.

    Attachments are repo-relative source paths, copied binary-safe
    (GAIA ships pdf/xlsx/mp3/png); missing ones are skipped aloud.
    """
    import shutil

    d = tempfile.mkdtemp(prefix="eval-task-")
    for rel, content in (files or {}).items():
        full = os.path.join(d, rel)
        os.makedirs(os.path.dirname(full) or d, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    for src in attachments or []:
        src_full = os.path.join(src_root, src)
        if not os.path.isfile(src_full):
            print(f"attachment missing (skipped): {src}", flush=True)
            continue
        dst = os.path.join(d, os.path.basename(src))
        shutil.copyfile(src_full, dst)
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
