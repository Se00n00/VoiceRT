"""Sim-Agent: subagents simulated with ONE agent (no extra models/VRAM).

Simple turns flow exactly as today (direct ``run_text``). Complex turns
— detected by a todo-count gate, no classifier call — run as:

  plan (one cheap call, numbered lines) -> stash parent frame (persisted)
  -> per subtask: lean context (system sliver + subtask preamble +
     narrowed tools + scratch session) -> result envelope back to parent
  -> final answer composed from envelopes -> restore everything.

Depth is capped at 1: a subtask emitting its own plan runs flat, never
nested (a non-empty stash frame is an error, not recursion). Guards
(confirm_fn, timeouts, step budgets) pass through unchanged — subtask
actions face the identical y/n gate and policy as direct turns.
"""
import json
import os
import re
import tempfile
import time

from src.agent.events import AgentEvent
from src.models.llm import split_thinking
from src.prompts.sim import (  # noqa: F401  (single home, re-exported)
    FINAL_COMPOSE_TEMPLATE,
    MAX_SUBTASKS,
    PLAN_PROMPT,
    SUBTASK_ENVELOPE_TEMPLATE,
    SUBTASK_HEAD_TEMPLATE,
    SYSTEM_SLIVER,
)
from src.prompts.terminal import TERMINAL_PREAMBLE
from src.tools.terminal import TERMINAL_TOOLS, tools_for_request

__all__ = ["SimAgent", "parse_plan", "SYSTEM_SLIVER",
           "build_subtask_preamble", "MAX_SUBTASKS", "PLAN_PROMPT",
           "FINAL_COMPOSE_TEMPLATE", "SUBTASK_HEAD_TEMPLATE",
           "SUBTASK_ENVELOPE_TEMPLATE"]

def build_subtask_preamble(i: int, n: int, task: str, parent_goal: str,
                             allowed_ops: list, context_dump: str,
                             base_preamble: str) -> str:
    """Four-section subtask preamble (Goal / Return Format / Warnings /
    Context Dump). The ops catalog + parser contract still come from
    ``base_preamble`` unchanged — this header only frames the subtask."""
    ops = ", ".join(allowed_ops) if allowed_ops else "all"
    head = SUBTASK_HEAD_TEMPLATE % (
        i, n, task, parent_goal[:500], ops,
        (context_dump or "(none)")[:1500])
    return head + "\n\n" + base_preamble

_PLAN_RE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")


def parse_plan(text: str, cap: int = MAX_SUBTASKS) -> list:
    """Numbered lines -> subtask list (capped). Pure."""
    out = []
    for line in str(text or "").splitlines():
        m = _PLAN_RE.match(line)
        if m and m.group(2).strip():
            out.append(m.group(2).strip())
        if len(out) >= cap:
            break
    return out


class SimAgent:
    """Single-agent subagent simulator over a VoiceAgent."""

    def __init__(self, agent, max_subtasks: int = MAX_SUBTASKS):
        self.agent = agent
        self.max_subtasks = max(1, int(max_subtasks))

    # -- stash ------------------------------------------------------
    def _frame_path(self, sid: str) -> str:
        mem = getattr(getattr(self.agent, "config", None), "memory_dir",
                      "memory")
        return os.path.join(str(mem or "memory"), "sim_%s.json" % sid)

    def _load_frame(self, sid: str):
        try:
            with open(self._frame_path(sid), encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _save_frame(self, sid: str, frame: dict) -> None:
        try:
            path = self._frame_path(sid)
            os.makedirs(os.path.dirname(os.path.abspath(path)),
                        exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(frame, f)
            os.replace(tmp, path)
        except Exception:
            pass

    def _clear_frame(self, sid: str) -> None:
        try:
            os.unlink(self._frame_path(sid))
        except Exception:
            pass

    # -- run --------------------------------------------------------
    async def run_task(self, text: str, *, sid: str, cwd: str,
                       confirm_fn=None):
        """Direct run for simple tasks, simulated subagents for complex.

        Yields the same term/* events as ``run_text`` (plus sys notes).
        """
        agent = self.agent
        # Phase 0 — plan (one cheap tool-free call).
        try:
            plan_res = await agent.llm.generate(
                agent.llm.messages(PLAN_PROMPT % text, []),
                max_tokens=256)
            plan_raw = str(getattr(plan_res, "text", "") or "")
        except Exception as exc:
            yield AgentEvent(node="term", kind="error",
                             data={"message": "sim plan failed: %s" % exc
                                   [:200]})
            return
        try:
            _, plan_text = split_thinking(plan_raw)
        except Exception:
            plan_text = plan_raw
        subtasks = parse_plan(plan_text, self.max_subtasks)
        if len(subtasks) < 2:
            async for ev in agent.run_text(text, session_id=sid, cwd=cwd,
                                           confirm_fn=confirm_fn):
                yield ev
            return
        if self._load_frame(sid):
            yield AgentEvent(
                node="term", kind="error",
                data={"message": "sim stash occupied for %s: refusing "
                                 "nested simulation (depth 1 only)" % sid})
            return
        yield AgentEvent(node="term", kind="sys",
                         data={"message": "sim-agent: %d subtasks" %
                                          len(subtasks)})
        # Snapshot parent config (restore in finally).
        llm = agent.llm
        chat_model = agent.chat_model
        old_system = getattr(getattr(llm, "config", None),
                             "system_prompt", None)
        old_preamble = getattr(llm, "terminal_preamble", None)
        old_bound = getattr(chat_model, "bound_tools", None)
        frame = {"sid": sid, "goal": text, "todos": subtasks,
                 "results": [], "at": time.time(), "status": "running"}
        self._save_frame(sid, frame)
        try:
            by_name = {t["name"]: t for t in TERMINAL_TOOLS}
            for i, task in enumerate(subtasks):
                scratch = "%s:sub%d" % (sid, i)
                # Lean context: sliver + subtask preamble + narrowed tools
                # + scratch session (parent history untouched).
                try:
                    llm.config = llm.config.model_copy(
                        update={"system_prompt": SYSTEM_SLIVER % text[:500]})
                except Exception:
                    pass
                try:
                    routed = tools_for_request(task, "") or list(by_name)
                    names = [t.get("name") if isinstance(t, dict) else t
                             for t in routed]
                    allowed = [n for n in names if n in by_name]
                    chat_model.bound_tools = [by_name[n] for n in allowed]
                except Exception:
                    allowed = list(by_name)
                try:
                    parent_hist = agent.sessions.history(sid) or []
                    dump = "\n".join(
                        "%s: %s" % (m.get("role", "?"),
                                    str(m.get("content", ""))[:400])
                        for m in parent_hist[-4:])
                except Exception:
                    dump = ""
                llm.terminal_preamble = build_subtask_preamble(
                    i + 1, len(subtasks), task, text, allowed, dump,
                    self._base_preamble())
                reply, error = "", ""
                try:
                    async for ev in agent.run_text(
                            task, session_id=scratch, cwd=cwd,
                            confirm_fn=confirm_fn):
                        yield ev
                        if ev.kind == "chat":
                            reply = str((ev.data or {}).get("reply", ""))
                        if ev.kind == "error":
                            error = str((ev.data or {}).get("message", ""))
                except Exception as exc:
                    error = str(exc)[:200]
                result = reply or ("failed: %s" % error if error else "empty")
                frame["results"].append({"task": task, "result": result})
                self._save_frame(sid, frame)
                envelope = SUBTASK_ENVELOPE_TEMPLATE % (i + 1, len(subtasks), task, result)
                try:
                    agent.sessions.append(sid, "assistant", envelope)
                except Exception:
                    pass
                yield AgentEvent(node="term", kind="sys",
                                 data={"message": envelope[:300]})
            frame["status"] = "done"
            self._save_frame(sid, frame)
            final_q = FINAL_COMPOSE_TEMPLATE % text
            async for ev in agent.run_text(final_q, session_id=sid, cwd=cwd,
                                           confirm_fn=confirm_fn):
                yield ev
        finally:
            try:
                if old_system is not None:
                    llm.config = llm.config.model_copy(
                        update={"system_prompt": old_system})
            except Exception:
                pass
            try:
                llm.terminal_preamble = old_preamble
            except Exception:
                pass
            try:
                chat_model.bound_tools = old_bound
            except Exception:
                pass
            if (frame.get("status") == "done"
                    and not any("failed" in str(r.get("result", ""))
                                for r in frame.get("results", []))):
                self._clear_frame(sid)

    @staticmethod
    def _base_preamble() -> str:
        try:
            return TERMINAL_PREAMBLE
        except Exception:
            return ""
