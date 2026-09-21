"""Terminal harness: single-model voice agent for the shell, as LangGraph.

Same contract as the browser tools: the ONE Qwen model either chats
(plain text, spoken back) or emits one JSON ``TerminalAction`` per step.
The turn is a compiled :class:`StateGraph`::

    propose -> {speak | gate | } ; gate -> {exec | speak}
    exec -> propose (loop, max_steps) ; speak -> END

- ``propose``: ``LlmModel.messages_for_terminal`` + ``generate`` + parse.
- ``gate``: :func:`check_policy` deny/confirm/allow; confirms call the
  injected ``confirm_fn`` (the TUI prompts ``y/n``; default denies).
- ``exec``: real subprocess/file ops with timeout + output cap.
- ``speak``: TTS the final reply (or denial), remember the turn in the
  LangChain session memory, emit the turn summary.

No checkpointer: memory lives in ``LangChainSessionMemory`` (RAM-only),
same as the voice loop. Models arrive via ``functools.partial`` binding
in :func:`build_terminal_graph`, so nodes stay unit-testable with fakes.
"""
import asyncio
import os
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.events import AgentEvent
from src.agent.memory import LangChainSessionMemory
from src.tools.terminal import (
    TerminalAction,
    check_policy,
    parse_terminal_action,
)

__all__ = [
    "TerminalConfig",
    "TerminalState",
    "build_terminal_graph",
    "TerminalHarness",
    "run_command",
]

from typing import TypedDict


class TerminalState(TypedDict, total=False):
    text: str              # user request (typed or STT)
    sid: str | None        # session id (None = stateless)
    history: list          # session messages for the prompt
    steps: list            # [{action, observation}]
    reply: str             # final assistant text
    done: bool             # loop exit flag
    cwd: str               # working directory for exec
    gate_ok: bool          # policy gate passed (transient)
    node_s: dict           # per-node seconds


@dataclass(frozen=True)
class TerminalConfig:
    """Per-harness knobs. Not YAML — construct in code."""

    max_steps: int = 6
    max_tokens_per_step: int = 64
    cwd: str = "."
    timeout_s: float = 30.0
    out_cap: int = 6000


def _writer(provided=None):
    if provided is not None:
        return provided
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return lambda event: None


def run_command(action: TerminalAction, cwd: str = ".",
                timeout_s: float = 30.0, out_cap: int = 6000) -> str:
    """Execute one validated action. No weights. Never raises."""
    try:
        if action.op == "exec":
            p = subprocess.run(
                action.command, shell=True, cwd=cwd or ".",
                capture_output=True, text=True, timeout=timeout_s)
            out = (p.stdout or "") + (p.stderr or "")
            if len(out) > out_cap:
                out = out[:out_cap] + f"\n…[truncated {len(out)} chars]"
            return f"rc={p.returncode}\n{out.strip() or '(no output)'}"
        if action.op == "list":
            target = action.path.strip() or "."
            base = os.path.abspath(cwd or ".")
            full = os.path.abspath(os.path.join(base, target))
            if not full.startswith(base):
                return "denied: path escapes cwd"
            try:
                names = sorted(os.listdir(full))
            except Exception as exc:
                return f"error: {exc}"
            return "\n".join(names[:200]) or "(empty)"
        if action.op == "read":
            base = os.path.abspath(cwd or ".")
            full = os.path.abspath(os.path.join(base, action.path.strip()))
            if not full.startswith(base):
                return "denied: path escapes cwd"
            try:
                with open(full, "r", errors="replace") as f:
                    data = f.read(out_cap + 1)
            except Exception as exc:
                return f"error: {exc}"
            if len(data) > out_cap:
                data = data[:out_cap] + "\n…[truncated]"
            return data or "(empty file)"
        if action.op == "write":
            base = os.path.abspath(cwd or ".")
            full = os.path.abspath(os.path.join(base, action.path.strip()))
            if not full.startswith(base):
                return "denied: path escapes cwd"
            try:
                os.makedirs(os.path.dirname(full) or base, exist_ok=True)
                with open(full, "w") as f:
                    f.write(action.text)
            except Exception as exc:
                return f"error: {exc}"
            return f"wrote {len(action.text)} chars to {action.path.strip()}"
        return f"unknown op: {action.op}"
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s:.0f}s"
    except Exception as exc:  # never break the graph
        return f"error: {exc}"


def _is_degenerate(raw: str) -> bool:
    """Small-model babble guard: one word dominating a long reply."""
    words = str(raw or "").split()
    if len(words) < 10:
        return False
    from collections import Counter

    top = Counter(w.lower() for w in words).most_common(1)[0][1]
    return top / len(words) > 0.4


async def _generate_once(llm, messages, limit: int) -> str:
    res = await llm.generate(messages, max_tokens=limit)
    return str(getattr(res, "text", "") or "")


async def propose_node(state, *, llm, cfg: TerminalConfig, writer=None):
    """One LLM call with the shared model; append a step or finish."""
    w = _writer(writer)
    steps = list(state.get("steps") or [])
    obs = ""
    if steps:
        last = steps[-1]
        # Tight cap: prompt+max_tokens must stay under the 512-token
        # context on both backends (paged admission + fused KV cache).
        # Truncated HERE (not just in messages_for_terminal) so any
        # llm-like object stays within budget.
        obs = str(last.get("observation", ""))[:350]
    text = str(state.get("text", "") or "")[:500]
    messages = llm.messages_for_terminal(
        text, state.get("history") or [],
        cwd=state.get("cwd") or cfg.cwd, observation=obs)
    try:
        raw = await _generate_once(llm, messages, cfg.max_tokens_per_step)
        if _is_degenerate(raw):
            # One retry with a nudge; small models sometimes self-correct.
            retry_msgs = messages + [{"role": "user", "content":
                "That reply was garbled. Answer again: ONLY one JSON action or one short sentence."}]
            raw = await _generate_once(llm, retry_msgs, cfg.max_tokens_per_step)
            if _is_degenerate(raw):
                w(AgentEvent(node="term", kind="error",
                             data={"message": "model output degenerate"}))
                return {"reply": "Sorry — I garbled that. Try rephrasing.", "done": True}
    except Exception as exc:
        w(AgentEvent(node="term", kind="error", data={"message": str(exc)[:300]}))
        return {"reply": f"model error: {exc}", "done": True}
    action = parse_terminal_action(raw)
    if action is None:
        w(AgentEvent(node="term", kind="chat", data={"reply": raw.strip()}))
        return {"reply": raw.strip(), "done": True}
    if action.op == "done":
        reply = action.reply or "Done."
        w(AgentEvent(node="term", kind="chat", data={"reply": reply}))
        return {"reply": reply, "done": True}
    w(AgentEvent(node="term", kind="action", data={"action": action.as_dict()}))
    steps.append({"action": action.as_dict(), "observation": ""})
    return {"steps": steps}


def route_after_propose(state) -> str:
    if state.get("done"):
        return "speak"
    steps = state.get("steps") or []
    if not steps:
        return "speak"
    return "gate"


async def gate_node(state, *, confirm_fn=None, writer=None):
    """Policy gate: deny -> finish; confirm -> ask; allow -> exec."""
    w = _writer(writer)
    steps = list(state.get("steps") or [])
    raw = (steps[-1]["action"] if steps else {})
    action = TerminalAction(op=raw.get("action", "done"),
                            command=raw.get("command", ""),
                            path=raw.get("path", ""),
                            text=raw.get("text", ""),
                            reply=raw.get("reply", ""))
    verdict, reason = check_policy(action)
    if verdict == "deny":
        w(AgentEvent(node="term", kind="deny",
                     data={"action": action.as_dict(), "reason": reason}))
        return {"reply": f"Blocked: {reason}.", "done": True}
    if verdict == "confirm":
        w(AgentEvent(node="term", kind="confirm",
                     data={"action": action.as_dict(), "reason": reason}))
        ok = False
        try:
            if confirm_fn is not None:
                ok = bool(await confirm_fn(action) if asyncio.iscoroutinefunction(confirm_fn)
                          else confirm_fn(action))
        except Exception:
            ok = False
        if not ok:
            w(AgentEvent(node="term", kind="deny",
                         data={"action": action.as_dict(), "reason": "denied by user"}))
            return {"reply": "Cancelled — I did not run that.", "done": True}
    return {"gate_ok": True}


def route_after_gate(state) -> str:
    if state.get("done"):
        return "speak"
    return "exec"


async def exec_node(state, *, cfg: TerminalConfig, writer=None):
    """Run the gated action, record the observation, loop or finish."""
    w = _writer(writer)
    t0 = time.perf_counter()
    steps = list(state.get("steps") or [])
    raw = (steps[-1]["action"] if steps else {})
    action = TerminalAction(op=raw.get("action", "done"),
                            command=raw.get("command", ""),
                            path=raw.get("path", ""),
                            text=raw.get("text", ""),
                            reply=raw.get("reply", ""))
    obs = await asyncio.to_thread(run_command, action,
                                  state.get("cwd") or cfg.cwd,
                                  cfg.timeout_s, cfg.out_cap)
    steps[-1] = {"action": action.as_dict(), "observation": obs}
    w(AgentEvent(node="term", kind="observation",
                 data={"action": action.as_dict(),
                       "observation": obs[:1000],
                       "exec_s": time.perf_counter() - t0}))
    if len(steps) >= cfg.max_steps:
        return {"steps": steps, "reply": "Ran out of steps. Last result:\n" + obs[:800],
                "done": True}
    if len(steps) >= 2 and steps[-1]["action"] == steps[-2]["action"]:
        # Weak models repeat the same call instead of 'done' — stop early
        # with the (identical) result rather than burning remaining steps.
        w(AgentEvent(node="term", kind="chat",
                     data={"reply": "Result:\n" + obs[:800]}))
        return {"steps": steps, "reply": "Result:\n" + obs[:800],
                "done": True}
    return {"steps": steps, "gate_ok": False}


def route_after_exec(state) -> str:
    return "speak" if state.get("done") else "propose"


def _speak_head(reply: str, limit: int = 280) -> str:
    """Head of a reply for voicing; full text stays on screen/in memory."""
    t = " ".join(str(reply or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    dot = cut.rfind(". ")
    head = (cut[:dot + 1] if dot > 120 else cut).strip()
    return f"{head} … full output is on screen."


async def speak_node(state, *, llm=None, tts=None, sessions=None, writer=None):
    """Speak the final reply, remember the turn, emit the summary.

    Only the head of long replies is voiced (full text stays in chat +
    memory): reading a multi-KB listing aloud is unlistenable and pure
    echo surface for the mic.
    """
    w = _writer(writer)
    reply = str(state.get("reply", "") or "")
    if tts is not None and reply:
        try:
            out = await tts.speak(_speak_head(reply))
            import numpy as _np

            w(AgentEvent(node="term", kind="audio", data={
                "wav": _np.asarray(out.wav, dtype="float32"),
                "sr": int(out.sample_rate), "sentence": reply[:500]}))
        except Exception as exc:
            w(AgentEvent(node="term", kind="error",
                         data={"message": f"tts failed: {exc}"[:200]}))
    if sessions is not None and state.get("sid") and (state.get("text") or reply):
        try:
            sessions.remember_turn(state["sid"], state.get("text", ""), reply)
        except Exception:
            pass
    w(AgentEvent(node="term", kind="summary", data={
        "text": state.get("text", ""), "reply": reply,
        "steps": state.get("steps") or [], "session_id": state.get("sid")}))
    return {"reply": reply, "done": True}


def build_terminal_graph(*, llm, tts=None, sessions=None,
                         config: TerminalConfig | None = None,
                         confirm_fn=None):
    """Compile the terminal turn graph (LangGraph StateGraph)."""
    cfg = config or TerminalConfig()
    builder = StateGraph(TerminalState)
    builder.add_node("propose", partial(propose_node, llm=llm, cfg=cfg))
    builder.add_node("gate", partial(gate_node, confirm_fn=confirm_fn))
    builder.add_node("exec", partial(exec_node, cfg=cfg))
    builder.add_node("speak", partial(speak_node, llm=llm, tts=tts,
                                      sessions=sessions))
    builder.add_edge(START, "propose")
    builder.add_conditional_edges("propose", route_after_propose,
                                  {"speak": "speak", "gate": "gate"})
    builder.add_conditional_edges("gate", route_after_gate,
                                  {"speak": "speak", "exec": "exec"})
    builder.add_conditional_edges("exec", route_after_exec,
                                  {"speak": "speak", "propose": "propose"})
    builder.add_edge("speak", END)
    return builder.compile()


class TerminalHarness:
    """One shared-model terminal agent: text in, actions + speech out."""

    def __init__(self, llm, tts=None, sessions=None,
                 config: TerminalConfig | None = None, confirm_fn=None):
        self.llm = llm
        self.tts = tts
        self.sessions = sessions or LangChainSessionMemory()
        self.config = config or TerminalConfig()
        self.confirm_fn = confirm_fn
        self._graph = None

    def _compiled(self):
        if self._graph is None:
            self._graph = build_terminal_graph(
                llm=self.llm, tts=self.tts, sessions=self.sessions,
                config=self.config, confirm_fn=self.confirm_fn)
        return self._graph

    async def run_turn(self, text: str, session_id: str | None = None,
                       cwd: str | None = None):
        """Run one turn, yielding :class:`AgentEvent` per step."""
        if not str(text or "").strip():
            raise ValueError("empty text")
        sid = session_id
        state_in: dict[str, Any] = {
            "text": str(text),
            "sid": sid,
            "history": (self.sessions.history(sid) if sid else []),
            "steps": [],
            "reply": "",
            "done": False,
            "cwd": cwd or self.config.cwd,
            "node_s": {},
        }
        async for event in self._compiled().astream(state_in, stream_mode="custom"):
            yield event
