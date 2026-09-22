"""VoiceAgent: ONE autonomous agent for voice, terminal, and everything else.

VAD -> STT -> LLM -> TTS as a single deepagents loop over the local model:

- VAD is always listening: transports keep endpointing audio into turns.
- speech -> STT text -> deep-agent turn (tools + JSON session memory) -> TTS.
- if a turn is already running for a session, new input is queued and
  injected as follow-up user message(s) once the active turn drains.
- every session persists to ``sessions/<sid>.json``.

Tools and prompts live in other files (``src/tools/terminal.py``,
``src/agent/chat_model.py``, ``src/agent/terminal.py``); this class wires
legs + agent + transports.

Calling an instance streams a voice turn (server.py ``/talk``)::

    agent = VoiceAgent()
    await agent.warm()
    async for event in agent(audio, sr=16000, session_id="..."):
        ...  # vad/stt/llm/tts events, then the turn summary

Text turns go through :meth:`run_text` (TUI, bridge, CLI)::

    async for event in agent.run_text("list files", session_id="...",
                                      cwd=".", confirm_fn=...):
        ...  # term events, audio, then the summary

Admission (FIFO queue + timeout) and the per-turn VRAM guard live here, so
every entry point is protected no matter the transport.
"""
import asyncio
import time
import types
from dataclasses import dataclass, field

import numpy as np

from src.agent.events import AgentEvent  # noqa: F401  (public currency)
from src.agent.memory import JsonSessionMemory
from engine import new_session_id
from src.models.llm import LlmConfig, LlmModel, SYSTEM_PROMPT, split_thinking
from src.models.runtime import FIFOScheduler, check_budget
from src.models.stt import SttConfig, SttModel
from src.models.tts import TtsConfig, TtsModel
from src.models.vad import VadConfig, VadModel
from src.agent.prompts import TERMINAL_PREAMBLE
from src.tools.terminal import (
    is_degenerate,
    parse_bare_tail,
    parse_terminal_action,
    parse_xml_action,
)

__all__ = ["VoiceAgentConfig", "VoiceAgent"]

# suffix appended to the shared tool preamble per modality. Voice replies
# must stay short enough to speak; text turns work until done.
_VOICE_SUFFIX = "\n\n" + SYSTEM_PROMPT
_TEXT_SUFFIX = ("\n\nWork until done: use tools step by step, then give one "
                "short final reply.")

_TRIM_PAD_S = 0.15


@dataclass(frozen=True)
class VoiceAgentConfig:
    """Whole-agent config. Legs are dataclasses; no YAML anywhere."""

    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    max_audio_s: float = 60.0
    max_session_turns: int = 20
    session_ttl_s: float = 1800.0
    max_sessions: int = 1000
    sessions_dir: str = "sessions"
    max_inflight: int = 4
    queue_timeout_s: float = 10.0
    vram_budget_mb: float = 3800.0
    per_turn_mb: float = 150.0
    trim_pad_s: float = 0.15
    # deep-agent loop + tool execution knobs (were TerminalConfig).
    max_agent_steps: int = 6
    tool_timeout_s: float = 30.0
    tool_out_cap: int = 6000
    # Docker sandbox for tool execution (None = host, as before).
    sandbox: object = None
    # paged LLM engine (src/inference) — shares VRAM budget, uses real QwenRunner
    llm_paged: bool = False  # set True to route LLM via paged engine (recommended on CUDA)
    llm_paged_blocks: int = 16
    llm_paged_batch_size: int = 4


def _lc_messages(history: list) -> list:
    """Session dicts -> LangChain messages (user/assistant only)."""
    from langchain_core.messages import AIMessage, HumanMessage

    out = []
    for m in history or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), str(m.get("content", "") or "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
    return out


def _chunk_pieces(chunk) -> list[str]:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return [p for p in parts if p]
    return []


def _trim(audio, sr, segs, pad_s=_TRIM_PAD_S):
    wav = np.asarray(audio, dtype=np.float32).ravel()
    dur = len(wav) / float(sr)
    s0 = max(0.0, float(segs[0][0]) - pad_s)
    s1 = min(dur, float(segs[-1][1]) + pad_s)
    if s1 - s0 < 0.05 or (s0 <= 0.0 and s1 >= dur):
        return audio
    return wav[int(s0 * sr):int(s1 * sr)]


def _vram_mb():
    try:
        from src.models.runtime import max_allocated_mb

        return float(max_allocated_mb())
    except Exception:
        return None


class VoiceAgent:
    """VAD -> STT -> deep-agent LLM -> TTS, yielding every node's stream."""

    def __init__(self, config: VoiceAgentConfig | None = None):
        self.config = config or VoiceAgentConfig()
        cfg = self.config
        # if paged LLM requested, inject paged flags into llm config
        llm_cfg = cfg.llm
        if cfg.llm_paged:
            # rebuild llm config with paged enabled (frozen dataclass)
            from dataclasses import replace
            llm_cfg = replace(llm_cfg, use_paged=True,
                              paged_blocks=cfg.llm_paged_blocks,
                              paged_batch_size=cfg.llm_paged_batch_size)
        self.vad = VadModel(cfg.vad)
        self.stt = SttModel(cfg.stt)
        self.llm = LlmModel(llm_cfg)
        self.tts = TtsModel(cfg.tts)
        # the langchain face of the local model: every agent turn below
        # runs through this (deepagents calls bind_tools/stream on it).
        from src.agent.chat_model import LocalChatModel

        self.chat_model = LocalChatModel(llm=self.llm)
        self.sessions = JsonSessionMemory(
            sessions_dir=cfg.sessions_dir,
            max_turns=cfg.max_session_turns,
            max_age_s=cfg.session_ttl_s,
            max_sessions=cfg.max_sessions,
        )
        self._sched = FIFOScheduler(cfg.max_inflight)
        self.missing: list = []
        self._warmed = False
        # per-session turn serialization: at most one agent loop runs per
        # sid; contenders queue up and are injected as follow-up user
        # messages once the active turn drains.
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, list[str]] = {}
        # per-session approved command prefixes ("always" answers).
        self.approvals: dict[str, set[str]] = {}
        # paged engine handle (mirrors llm._paged_engine after warm)
        self._paged_engine = None

    @property
    def warmed(self) -> bool:
        return self._warmed

    async def warm(self) -> "VoiceAgent":
        """Build every leg; collect failures instead of raising."""
        self.missing = []
        for name in ("vad", "stt", "llm", "tts"):
            try:
                await getattr(self, name).warm()
            except Exception as exc:  # noqa: BLE001 - omit-and-report
                self.missing.append(f"{name} leg: {exc}")
        # expose paged engine if llm was built with it
        try:
            eng = None
            if hasattr(self.llm, "_paged_engine"):
                cand = getattr(self.llm, "_paged_engine")
                eng = cand() if callable(cand) else cand
                # also check inst attribute
                if eng is None and hasattr(self.llm, "_paged_engine_inst"):
                    eng = getattr(self.llm, "_paged_engine_inst")
            self._paged_engine = eng
            if self._paged_engine is None and getattr(self.config, "llm_paged", False):
                try:
                    self._paged_engine = self.llm._paged_engine()  # type: ignore
                except Exception:
                    pass
        except Exception:
            self._paged_engine = None
        self._warmed = True
        return self

    @property
    def paged_engine(self):
        # lazy fallback — only after warm to avoid eager weight fetch in tests
        if not self._warmed:
            return self._paged_engine
        if self._paged_engine is None:
            try:
                eng = self.llm._paged_engine()  # type: ignore
                self._paged_engine = eng
            except Exception:
                pass
        return self._paged_engine

    def engine_stats(self):
        """Paged engine stats if active, else None."""
        eng = self.paged_engine
        if eng is None:
            return None
        try:
            return eng.stats()  # type: ignore
        except Exception:
            return None

    def remember_approval(self, session_id: str, command: str):
        """Remember that this session approved a command prefix."""
        if not session_id or not command:
            return
        self.approvals.setdefault(session_id, set()).add(command.strip().split()[0][:40])

    def is_approved(self, session_id: str | None, action) -> bool:
        if not session_id:
            return False
        allow = self.approvals.get(session_id, set())
        if not allow:
            return False
        cmd = getattr(action, "command", "") or ""
        first = cmd.strip().split()[0] if cmd.strip() else ""
        return first in allow or action.op in allow

    # -- admission ----------------------------------------------------
    async def _admit(self):
        ticket = await asyncio.to_thread(
            self._sched.acquire, True, self.config.queue_timeout_s)
        if ticket is None:
            raise TimeoutError(
                f"server saturated ({self._sched.max_concurrency} in flight, "
                f"queue waited {self.config.queue_timeout_s:.0f}s); retry")
        return ticket

    # -- agent construction -------------------------------------------
    def _open_shell(self, cwd: str):
        """Persistent shell for one turn (container if sandbox configured)."""
        sandbox = None
        if getattr(self.config, "sandbox", None) is not None:
            from src.sandbox.docker import DockerShell

            sandbox = self.config.sandbox.create(host_cwd=cwd)
            sandbox.ensure_running()  # may raise DockerSandboxError
            shell = DockerShell(sandbox, timeout_s=self.config.tool_timeout_s)
        else:
            from src.agent.shell import PersistentShell

            shell = PersistentShell(cwd=cwd, timeout_s=self.config.tool_timeout_s)
        return shell, sandbox

    def _close_shell(self, shell, sandbox) -> None:
        try:
            shell.close()
        except Exception:
            pass
        if sandbox is not None:
            try:
                sandbox.close()
            except Exception:
                pass

    def _tool_cfg(self):
        cfg = self.config
        return types.SimpleNamespace(timeout_s=cfg.tool_timeout_s,
                                     out_cap=cfg.tool_out_cap)

    def _build_deep_agent(self, ctx, system_suffix: str):
        from deepagents import create_deep_agent
        from langchain.agents.middleware import TodoListMiddleware

        from src.agent.prompts import TERMINAL_PREAMBLE
        from src.agent.terminal import build_terminal_tools

        return create_deep_agent(
            model=self.chat_model,
            tools=build_terminal_tools(ctx),
            middleware=[TodoListMiddleware()],
            system_prompt=TERMINAL_PREAMBLE + system_suffix,
        )

    def _confirm_wrapper(self, confirm_fn, sid: str | None):
        """Wrap confirm_fn with "always" -> remember approval."""
        async def _wrapper(action):
            res = None
            if confirm_fn is not None:
                res = (await confirm_fn(action)
                       if asyncio.iscoroutinefunction(confirm_fn)
                       else confirm_fn(action))
            if isinstance(res, str) and res.lower() in ("always", "a", "allowlist"):
                if sid:
                    self.remember_approval(sid, getattr(action, "command", "") or getattr(action, "op", ""))
                return "always"
            return bool(res) if res is not None else False

        return _wrapper

    # -- shared turn machinery ----------------------------------------
    def _norm_msg(self, m):
        if isinstance(m, dict):
            return (m.get("type") or m.get("role") or "",
                    m.get("content", ""), m.get("tool_calls"),
                    m.get("additional_kwargs") or {})
        return (getattr(m, "type", None) or "",
                getattr(m, "content", ""),
                getattr(m, "tool_calls", None),
                getattr(m, "additional_kwargs", None) or {})

    @staticmethod
    def _think_of(ak) -> str:
        return str(ak.get("thinking", "") or "") if isinstance(ak, dict) else ""

    @staticmethod
    def _resolve_reply(content: str) -> str:
        """Text-only AI content -> reply; legacy done envelopes resolve."""
        txt = str(content or "").strip()
        if not txt:
            return ""
        for cand in (parse_terminal_action(txt), parse_xml_action(txt),
                     parse_bare_tail(txt)):
            if cand is not None and cand.op == "done":
                return cand.reply or ""
        return txt

    async def _agent_invoke(self, text: str, *, sid: str | None,
                            cwd: str, confirm_fn, system_suffix: str,
                            out_turns: list):
        """One deep-agent invocation.

        Yields term/* mid-turn events (token/thinking/action/observation/
        confirm/deny/error). Appends ``(text, reply, thinking)`` per
        invocation to ``out_turns`` and remembers each exchange. The
        caller speaks replies and emits summaries.
        """
        from langchain_core.messages import AIMessageChunk, HumanMessage

        from src.agent.terminal import TurnCtx, _approval_key
        from src.models.llm import split_thinking

        try:
            shell, sandbox = self._open_shell(cwd)
        except Exception as exc:
            yield AgentEvent(node="term", kind="error",
                             data={"message": f"sandbox: {exc}"[:300]})
            return
        confirm = self._confirm_wrapper(confirm_fn, sid)
        ctx = TurnCtx(shell=shell, cwd=cwd,
                      approvals=set(self.approvals.get(sid, set())) if sid else set(),
                      confirm_fn=confirm,
                      loop=asyncio.get_running_loop(), cfg=self._tool_cfg(),
                      sandbox=sandbox)
        # seed "always" persistence back into the harness store
        _orig_confirm = confirm

        async def _seeded_confirm(action):
            res = await _orig_confirm(action)
            if res == "always" and sid:
                ctx.approvals.add(_approval_key(action))
            return res

        ctx.confirm_fn = _seeded_confirm
        try:
            agent = self._build_deep_agent(ctx, system_suffix)
        except Exception as exc:
            self._close_shell(shell, sandbox)
            yield AgentEvent(node="term", kind="error",
                             data={"message": f"agent build: {exc}"[:300]})
            return
        history = self.sessions.history(sid) if sid else []
        lc_msgs = _lc_messages(history) + [
            HumanMessage(content=str(text) + f"\nCWD: {cwd} SHELL: bash")]
        final_reply = ""
        thinking = ""
        last_think = ""
        did_work = False
        drained = 0

        def _drain():
            nonlocal drained
            while drained < len(ctx.sink):
                kind, data = ctx.sink[drained]
                drained += 1
                yield AgentEvent(node="term", kind=kind, data=data)

        rl = 10 + int(getattr(self.config, "max_agent_steps", 6) or 6) * 5
        try:
            stream = agent.astream({"messages": lc_msgs},
                                   config={"recursion_limit": rl},
                                   stream_mode=["messages", "updates"])
            async for chunk in stream:
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    mode, payload = chunk
                else:
                    mode, payload = "updates", chunk
                if mode == "messages":
                    msg, _meta = payload if isinstance(payload, (tuple, list)) else (payload, {})
                    if isinstance(msg, AIMessageChunk):
                        for piece in _chunk_pieces(msg):
                            yield AgentEvent(node="term", kind="token",
                                             data={"piece": piece[:500]})
                    continue
                if not isinstance(payload, dict):
                    continue
                for ev in _drain():
                    yield ev
                for _node, data in payload.items():
                    msgs = data.get("messages", []) if isinstance(data, dict) else []
                    for m in msgs if isinstance(msgs, list) else []:
                        role, content, tool_calls, ak = self._norm_msg(m)
                        if isinstance(content, list):
                            content = " ".join(
                                str(c.get("text", c)) for c in content
                                if isinstance(c, dict))
                        if role in ("ai", "assistant"):
                            think = self._think_of(ak)
                            if think and think != last_think:
                                last_think = think
                                thinking = think
                                yield AgentEvent(node="term", kind="thinking",
                                                 data={"text": think[:2000]})
                            if tool_calls:
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
                                        node="term", kind="action",
                                        data={"action": {"action": name, **args}})
                                    did_work = True
                            elif str(content or "").strip():
                                final_reply = self._resolve_reply(content)
                        elif role == "tool":
                            did_work = True
                            obs = str(content or "")
                            if obs.startswith("Blocked:"):
                                yield AgentEvent(
                                    node="term", kind="deny",
                                    data={"reason": obs[len("Blocked:"):].strip()[:300]})
                            elif obs.startswith("Cancelled"):
                                yield AgentEvent(
                                    node="term", kind="deny",
                                    data={"reason": "denied by user"})
                            else:
                                yield AgentEvent(
                                    node="term", kind="observation",
                                    data={"observation": obs[:1200]})
            for ev in _drain():
                yield ev
        except Exception as exc:  # never kill the turn on engine errors
            yield AgentEvent(node="term", kind="error",
                             data={"message": str(exc)[:300]})
        finally:
            self._close_shell(shell, sandbox)

        # split thinking (never voiced/memorized — only the answer is)
        thinking2, reply = split_thinking(final_reply)
        if thinking2 and thinking2 != last_think:
            thinking = thinking2
            yield AgentEvent(node="term", kind="thinking",
                             data={"text": thinking[:2000]})
        if not reply and did_work:
            reply = "Done."
        if reply and is_degenerate(reply):
            yield AgentEvent(node="term", kind="error",
                             data={"message": "model output degenerate"})
            reply = "Sorry — I garbled that. Try rephrasing."
        if sid and (text or reply):
            try:
                self.sessions.remember_turn(sid, str(text), reply)
            except Exception:
                pass
        out_turns.append((str(text), reply, thinking))

    async def _run_locked(self, text: str, *, sid: str | None, cwd: str,
                          confirm_fn, system_suffix: str, out_turns: list):
        """Serialize agent turns per session; queue + inject contenders.

        A second message arriving while ``sid`` has an active turn is
        appended to the pending queue (yielding one ``queued`` event) and
        injected as follow-up user message(s) once the active turn drains.
        """
        if sid is None:
            async for ev in self._agent_invoke(
                    text, sid=None, cwd=cwd, confirm_fn=confirm_fn,
                    system_suffix=system_suffix, out_turns=out_turns):
                yield ev
            return
        lock = self._locks.setdefault(sid, asyncio.Lock())
        if lock.locked():
            pend = self._pending.setdefault(sid, [])
            pend.append(str(text))
            yield AgentEvent(node="term", kind="queued",
                             data={"position": len(pend), "session_id": sid})
            return
        await lock.acquire()
        try:
            texts = self._pending.pop(sid, []) + [str(text)]
            while texts:
                t = texts.pop(0)
                async for ev in self._agent_invoke(
                        t, sid=sid, cwd=cwd, confirm_fn=confirm_fn,
                        system_suffix=system_suffix, out_turns=out_turns):
                    yield ev
                texts.extend(self._pending.pop(sid, []))
        finally:
            lock.release()

    # -- text turns (TUI, bridge, CLI) ----------------------------------
    async def run_text(self, text: str, session_id: str | None = None,
                       cwd: str | None = None, confirm_fn=None):
        """Run one text turn, yielding term/audio/summary events."""
        from src.agent.terminal import _speak_head

        if not str(text or "").strip():
            raise ValueError("empty text")
        sid = session_id
        base = cwd or "."
        turns: list = []
        last_reply, last_think = "", ""
        async for ev in self._run_locked(
                str(text), sid=sid, cwd=base, confirm_fn=confirm_fn,
                system_suffix=_TEXT_SUFFIX, out_turns=turns):
            yield ev
        for (_t, reply, thinking) in turns:
            if not reply:
                last_reply, last_think = reply, thinking
                continue
            if is_degenerate(reply):
                yield AgentEvent(node="term", kind="error",
                                 data={"message": "model output degenerate"})
                reply = "Sorry — I garbled that. Try rephrasing."
            last_reply, last_think = reply, thinking
            yield AgentEvent(node="term", kind="chat", data={"reply": reply})
            if self.tts is not None:
                try:
                    out = await self.tts.speak(_speak_head(reply))
                    import numpy as _np

                    yield AgentEvent(node="term", kind="audio", data={
                        "wav": _np.asarray(out.wav, dtype="float32"),
                        "sr": int(out.sample_rate), "sentence": reply[:500]})
                except Exception as exc:
                    yield AgentEvent(node="term", kind="error",
                                     data={"message": f"tts failed: {exc}"[:200]})
        yield AgentEvent(node="term", kind="summary", data={
            "text": str(text), "reply": last_reply,
            "thinking": (last_think or "")[:2000], "session_id": sid})

    # -- voice turns (server /talk) ---------------------------------------
    async def __call__(self, audio, sr: int = 16000, session_id: str | None = None):
        """Run one voice turn, yielding vad/stt/llm/tts/term events + summary."""
        from engine import SentenceSplitter

        wav = np.asarray(audio, dtype=np.float32).ravel()
        if wav.size == 0:
            raise ValueError("empty audio")
        dur_s = len(wav) / float(sr)
        if dur_s > self.config.max_audio_s:
            raise ValueError(
                f"audio {dur_s:.1f}s exceeds {self.config.max_audio_s:.0f}s cap")
        check_budget(self.config.per_turn_mb, self.config.vram_budget_mb,
                     what="voice turn")

        ticket = await self._admit()
        try:
            sid = session_id or new_session_id()
            remember = bool(session_id)
            t0 = time.perf_counter()
            node_s: dict = {}

            # -- VAD: speech spans (+ STT-window trim) -------------------
            t_vad = time.perf_counter()
            try:
                segs = await self.vad.segments(wav, int(sr))
            finally:
                node_s["vad"] = node_s.get("vad", 0.0) + time.perf_counter() - t_vad
            segments = [[float(a), float(b)] for a, b in segs.segments]
            audio = (wav if not segments
                     else _trim(wav, sr, segments, self.config.trim_pad_s))
            yield AgentEvent(node="vad", kind="segments", data={
                "segments": segments,
                "audio_dur_s": float(getattr(segs, "audio_dur_s", dur_s)),
                "speech_s": float(getattr(segs, "speech_s", 0.0) or 0.0),
            })
            if not segments:
                yield AgentEvent(node="stt", kind="text",
                                 data={"text": "", "silent": True})
                total = time.perf_counter() - t0
                yield AgentEvent(node="turn", kind="summary", data={
                    "text": "", "reply": "", "segments": segments,
                    "node_s": dict(node_s), "ttfa_s": total, "total_s": total,
                    "vram_mb": _vram_mb(), "session_id": sid})
                return

            # -- STT ------------------------------------------------------
            t_stt = time.perf_counter()
            try:
                res = await self.stt.transcribe(audio, int(sr))
            finally:
                node_s["stt"] = node_s.get("stt", 0.0) + time.perf_counter() - t_stt
            text = str(res.text)
            yield AgentEvent(node="stt", kind="text", data={
                "text": text, "rtf": float(res.rtf),
                "ttfs": float(res.ttfs), "dur_s": float(res.dur_s),
            })
            if not text.strip():
                total = time.perf_counter() - t0
                yield AgentEvent(node="turn", kind="summary", data={
                    "text": text, "reply": "", "segments": segments,
                    "node_s": dict(node_s), "ttfa_s": total, "total_s": total,
                    "vram_mb": _vram_mb(), "session_id": sid})
                return

            # -- LLM: the unified deep-agent turn (tools + memory) --------
            t_llm = time.perf_counter()
            first_audio_at = None
            first_piece = True
            turns: list = []
            eff_sid = sid if remember else None
            async for ev in self._run_locked(
                    text, sid=eff_sid, cwd=".", confirm_fn=None,
                    system_suffix=_VOICE_SUFFIX, out_turns=turns):
                if ev.kind == "token":
                    yield AgentEvent(node="llm", kind="token", data={
                        "token": str(ev.data.get("piece", "") or ""),
                        "first": first_piece})
                    first_piece = False
                elif ev.kind == "thinking":
                    yield AgentEvent(node="llm", kind="thinking",
                                     data={"text": ev.data.get("text", "")})
                elif ev.kind in ("action", "observation", "confirm", "deny",
                                 "error", "queued"):
                    yield ev
                # chat/summary left to the tails below (voice speaks + done)
            node_s["llm"] = node_s.get("llm", 0.0) + time.perf_counter() - t_llm

            # -- TTS: speak every reply sentence as it completes -----------
            for (_t, reply, _th) in turns:
                if not (reply or "").strip():
                    continue
                splitter = SentenceSplitter()
                pending: list[str] = []

                async def _speak_sentence(sent):
                    nonlocal first_audio_at
                    t_tts = time.perf_counter()
                    try:
                        out = await self.tts.speak(sent)
                    finally:
                        node_s["tts"] = (node_s.get("tts", 0.0)
                                         + time.perf_counter() - t_tts)
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter() - t0
                    yield AgentEvent(node="tts", kind="audio", data={
                        "wav": np.asarray(out.wav, dtype=np.float32),
                        "sr": int(out.sample_rate),
                        "sentence": str(out.sentence or sent),
                        "synth_s": float(out.synth_s),
                    })

                # feed whole reply through the splitter, then flush
                async for audio_ev in _feed_sentences(splitter, reply, _speak_sentence):
                    yield audio_ev
                yield AgentEvent(node="llm", kind="done", data={"text": reply})

            total = time.perf_counter() - t0
            last_reply = turns[-1][1] if turns else ""
            yield AgentEvent(node="turn", kind="summary", data={
                "text": text, "reply": last_reply, "segments": segments,
                "node_s": dict(node_s),
                "ttfa_s": first_audio_at if first_audio_at is not None else total,
                "total_s": total, "vram_mb": _vram_mb(), "session_id": sid})
        finally:
            self._sched.release(ticket)


async def _feed_sentences(splitter, reply: str, speak):
    """Push reply text through the splitter, speaking complete sentences."""
    for sent in splitter.push(reply):
        if (sent or "").strip():
            async for ev in speak(sent):
                yield ev
    tail = splitter.flush()
    if tail and (tail or "").strip():
        async for ev in speak(tail):
            yield ev
