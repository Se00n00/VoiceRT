"""One turn: box, graph build, stream -> AgentEvents, session write-back.

The runner owns the graph invocation and the event mapping. Session and
memory persistence are in middleware; this file only supplies the turn
identity to the builder and turns the stream into ``term/*`` events.
"""

import asyncio

import numpy as np
from langchain_core.messages import AIMessageChunk, HumanMessage

from src.agent.events import AgentEvent
from src.agent.long_running.box import TurnBoxError
from src.agent.long_running.messages import (
    chunk_pieces,
    finalize_reply,
    norm_msg,
    resolve_reply,
    text_of,
    think_of,
)
from src.agent.long_running.speech import speak_head

__all__ = ["TurnRunner"]


class TurnRunner:
    """Run text turns against the harness and yield agent events."""

    def __init__(self, host):
        """Bind the facade that owns config, box, worker and sessions."""
        self.host = host

    def _confirm_wrapper(self, confirm_fn, sid: str | None):
        """Adapt a raw confirm callback to allow/deny/allowlist semantics."""

        async def _wrapper(action):
            """Resolve one tool approval through the user callback."""
            res = None
            if confirm_fn is not None:
                res = (
                    await confirm_fn(action)
                    if asyncio.iscoroutinefunction(confirm_fn)
                    else confirm_fn(action)
                )
            if isinstance(res, str) and res.lower() in ("always", "a", "allowlist"):
                if sid:
                    command = ""
                    if isinstance(action, dict):
                        command = str(
                            action.get("command")
                            or (action.get("args") or {}).get("command")
                            or action.get("name")
                            or ""
                        )
                    else:
                        command = str(
                            getattr(action, "command", "")
                            or getattr(action, "op", "")
                        )
                    self.host.remember_approval(sid, command)
                return "always"
            return bool(res) if res is not None else False

        return _wrapper

    async def _agent_invoke(
        self,
        text: str,
        *,
        sid: str | None,
        cwd: str,
        confirm_fn,
        out_turns: list,
    ):
        """One deep-agent invocation, yielding term/* mid-turn events.

        Runs the full graph; the box owns the backend lifecycle.
        ..
        """
        raw_confirm = self._confirm_wrapper(confirm_fn, sid)

        async def _policy_confirm(name, args):
            return await raw_confirm({"name": name, "args": dict(args or {})})

        try:
            async with self.host.box.open(cwd) as (turn_cwd, turn_backend):
                async for ev in self._invoke_boxed(
                    text,
                    sid=sid,
                    cwd=turn_cwd,
                    backend=turn_backend,
                    confirmer=_policy_confirm,
                    out_turns=out_turns,
                ):
                    yield ev
        except TurnBoxError as exc:
            yield AgentEvent(
                node="term",
                kind="error",
                data={"message": f"sandbox: {exc}"[:300]},
            )

    async def _invoke_boxed(
        self,
        text: str,
        *,
        sid: str | None,
        cwd: str,
        backend,
        confirmer,
        out_turns: list,
    ):
        """Boxed worker invocation: build the graph and stream events.

        The caller owns the box lifecycle. Session history, memory and
        turn context are injected by middleware, so the graph starts
        with the current user message only.
        ..
        """
        try:
            agent = self.host.build_longrunning_agent(
                confirmer=confirmer,
                backend=backend,
                session_id=sid,
                text=str(text),
                cwd=cwd,
            )
            self.host.agent = agent
        except Exception as exc:
            yield AgentEvent(
                node="term", kind="error", data={"message": f"agent build: {exc}"[:300]}
            )
            return
        lc_msgs = [HumanMessage(content=str(text))]
        final_reply = ""
        thinking = ""
        last_think = ""
        live_think_shown = False
        did_work = False
        rl = 10 + int(getattr(self.host.config, "max_agent_steps", 6) or 6) * 5
        try:
            stream = agent.astream(
                {"messages": lc_msgs},
                config={"recursion_limit": rl},
                stream_mode=["messages", "updates"],
            )
            async for chunk in stream:
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    mode, payload = chunk
                else:
                    mode, payload = "updates", chunk
                if mode == "messages":
                    msg, _meta = (
                        payload if isinstance(payload, (tuple, list)) else (payload, {})
                    )
                    if isinstance(msg, AIMessageChunk):
                        for piece in chunk_pieces(msg):
                            yield AgentEvent(
                                node="term", kind="token", data={"piece": piece[:500]}
                            )
                        ak = getattr(msg, "additional_kwargs", None) or {}
                        if isinstance(ak, dict) and ak.get("thinking_delta"):
                            live_think_shown = True
                            yield AgentEvent(
                                node="term",
                                kind="thinking",
                                data={
                                    "text": str(ak["thinking_delta"])[:500],
                                    "append": True,
                                },
                            )
                    continue
                if not isinstance(payload, dict):
                    continue
                for _node, data in payload.items():
                    msgs = data.get("messages", []) if isinstance(data, dict) else []
                    for m in msgs if isinstance(msgs, list) else []:
                        role, content, tool_calls, ak = norm_msg(m)
                        content = text_of(content)
                        if role in ("ai", "assistant"):
                            think = think_of(ak)
                            if think and think != last_think:
                                last_think = think
                                thinking = think
                                if live_think_shown:
                                    live_think_shown = False
                                else:
                                    yield AgentEvent(
                                        node="term",
                                        kind="thinking",
                                        data={"text": think[:2000]},
                                    )
                            if tool_calls:
                                live_think_shown = False
                                for tc in tool_calls:
                                    if isinstance(tc, dict):
                                        name = tc.get("name", "?")
                                        args = tc.get("args") or {}
                                    else:
                                        name = getattr(tc, "name", "?")
                                        args = getattr(tc, "args", {}) or {}
                                    if not isinstance(args, dict):
                                        args = {}
                                    yield AgentEvent(
                                        node="term",
                                        kind="action",
                                        data={"action": {"action": name, **args}},
                                    )
                                    did_work = True
                            elif str(content or "").strip():
                                live_think_shown = False
                                final_reply = resolve_reply(content)
                        elif role == "tool":
                            live_think_shown = False
                            did_work = True
                            obs = str(content or "")
                            if obs.startswith("Blocked:"):
                                yield AgentEvent(
                                    node="term",
                                    kind="deny",
                                    data={
                                        "reason": obs[len("Blocked:") :].strip()[:300]
                                    },
                                )
                            elif obs.startswith("Cancelled"):
                                yield AgentEvent(
                                    node="term",
                                    kind="deny",
                                    data={"reason": "denied by user"},
                                )
                            else:
                                yield AgentEvent(
                                    node="term",
                                    kind="observation",
                                    data={"observation": obs[:1200]},
                                )
        except Exception as exc:
            yield AgentEvent(
                node="term", kind="error", data={"message": str(exc)[:300]}
            )
        result = finalize_reply(final_reply, did_work)
        if result.thinking and result.thinking != last_think:
            thinking = result.thinking
            if not live_think_shown:
                yield AgentEvent(
                    node="term", kind="thinking", data={"text": thinking[:2000]}
                )
        if result.degenerate:
            yield AgentEvent(
                node="term", kind="error", data={"message": "model output degenerate"}
            )
        out_turns.append((str(text), result.reply, thinking))

    async def _run_turn(
        self,
        text: str,
        *,
        sid: str | None,
        cwd: str,
        confirm_fn,
        out_turns: list,
    ):
        """Run one turn on the worker graph.

        Every turn runs the full agent; the reply is what the user hears.
        ..
        """
        reason = await self.host.worker.ensure()
        if reason:
            out_turns.append(self.host.worker.down_notice(str(text), reason))
            return
        before = len(out_turns)
        async for ev in self._agent_invoke(
            str(text),
            sid=sid,
            cwd=cwd,
            confirm_fn=confirm_fn,
            out_turns=out_turns,
        ):
            yield ev
        produced = any(str(reply).strip() for (_t, reply, _th) in out_turns[before:])
        if not produced:
            why = self.host.worker.missing_reason()
            if why:
                notice = self.host.worker.down_notice(str(text), why)
                if len(out_turns) > before:
                    out_turns[-1] = notice
                else:
                    out_turns.append(notice)

    async def run(
        self,
        text: str,
        *,
        sid: str | None,
        cwd: str,
        confirm_fn,
        out_turns: list,
    ):
        """Serialize and run one queued text through the turn pipeline."""

        async def _one(t: str):
            """Run one queued text through the turn pipeline."""
            async for ev in self._run_turn(
                t,
                sid=sid,
                cwd=cwd,
                confirm_fn=confirm_fn,
                out_turns=out_turns,
            ):
                yield ev

        async for ev in self.host.scheduler.run(sid, text, _one):
            yield ev

    async def run_text(
        self,
        text: str,
        session_id: str | None = None,
        cwd: str | None = None,
        confirm_fn=None,
        speak: bool = True,
    ):
        """Run one text turn, yielding term/audio/summary events."""
        if not str(text or "").strip():
            raise ValueError("empty text")
        sid = session_id
        turns: list = []
        last_reply, last_think = "", ""
        async for ev in self.run(
            str(text),
            sid=sid,
            cwd=cwd or ".",
            confirm_fn=confirm_fn,
            out_turns=turns,
        ):
            yield ev
        for _t, reply, thinking in turns:
            if not reply:
                last_reply, last_think = reply, thinking
                continue
            last_reply, last_think = reply, thinking
            yield AgentEvent(node="term", kind="chat", data={"reply": reply})
            if (
                speak
                and getattr(self.host.config, "speak_text_turns", True)
                and getattr(self.host, "tts", None) is not None
            ):
                try:
                    out = await self.host.tts.speak(speak_head(reply))
                    yield AgentEvent(
                        node="term",
                        kind="audio",
                        data={
                            "wav": np.asarray(out.wav, dtype="float32"),
                            "sr": int(out.sample_rate),
                            "sentence": reply[:500],
                        },
                    )
                except Exception as exc:
                    yield AgentEvent(
                        node="term",
                        kind="error",
                        data={"message": f"tts failed: {exc}"[:200]},
                    )
        yield AgentEvent(
            node="term",
            kind="summary",
            data={
                "text": str(text),
                "reply": last_reply,
                "thinking": (last_think or "")[:2000],
                "session_id": sid,
            },
        )

    async def collect_text(
        self,
        text: str,
        session_id: str | None = None,
        cwd: str | None = None,
        confirm_fn=None,
        speak: bool = True,
    ) -> str:
        """Run one text turn and return only the final reply."""
        reply = ""
        async for ev in self.run_text(
            text,
            session_id=session_id,
            cwd=cwd,
            confirm_fn=confirm_fn,
            speak=speak,
        ):
            if ev.kind == "chat":
                reply = str((ev.data or {}).get("reply", "") or "")
        return reply
