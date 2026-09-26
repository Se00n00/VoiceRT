"""VoiceAgent: ONE autonomous agent for voice, terminal, and everything else.

Device placement (Gemma era — LLM on CPU, fast hands on GPU):

- LLM  Gemma-4-E4B-it Q4_K_M via local llama.cpp sidecar (CPU-only,
  ``backend="gemma"``). Zero VRAM by design; the paged GPU engine is
  incoherent here and stays off.
- VAD  Silero ONNX (CPU, RTF ~0.01).
- STT  Whisper-base fused (GPU when CUDA is up, CPU fallback otherwise).
- TTS  Kokoro-82M (GPU when CUDA is up, CPU fallback otherwise).

Pipeline: speech -> VAD spans -> STT text -> deep-agent turn (tools +
JSON session memory) -> TTS. Text turns go through :meth:`run_text`
(TUI, bridge, CLI, evals); voice turns through :meth:`__call__`
(``server.py`` ``/talk``). If a turn is already running for a session,
new input is queued and injected as follow-up message(s) once the
active turn drains.
"""

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from src.agent.events import AgentEvent  # noqa: F401  (public currency)
from src.agent.memory import JsonSessionMemory
from src.models.llm import LlmConfig, LlmModel, SYSTEM_PROMPT, split_thinking
from src.models.stt import SttConfig, SttModel
from src.models.tts import TtsConfig, TtsModel
from src.models.vad import VadConfig, VadModel

__all__ = ["VoiceAgentConfig", "VoiceAgent"]

# suffix appended to the shared tool preamble per modality. Voice replies
# must stay short enough to speak; text turns work until done.
_VOICE_SUFFIX = "\n\n" + SYSTEM_PROMPT
_TEXT_SUFFIX = ("\n\nWork until done: use tools step by step, then give one "
                "short final reply.")

_TRIM_PAD_S = 0.15


def _speak_head(full: str, limit: int = 280) -> str:
    """Short speakable head of a reply (full text stays on screen)."""
    t = " ".join(str(full or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    dot = cut.rfind(". ")
    head = cut[: dot + 1] if dot > 120 else cut
    return head.strip() + " … full output is on screen."


@dataclass(frozen=True)
class VoiceAgentConfig:
    """Whole-agent config. Legs are dataclasses; no YAML anywhere."""

    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    max_audio_s: float = 60.0
    max_session_turns: int = 8
    session_ttl_s: float = 1800.0
    max_sessions: int = 1000
    sessions_dir: str = "sessions"
    max_inflight: int = 4
    queue_timeout_s: float = 10.0
    # VRAM budget covers the GPU legs only (STT + TTS). The LLM is a CPU
    # sidecar and holds zero VRAM — a sick GPU must never block a CPU turn.
    vram_budget_mb: float = 3800.0
    per_turn_mb: float = 150.0
    trim_pad_s: float = 0.15
    # deep-agent loop + tool execution knobs.
    max_agent_steps: int = 6
    recursion_limit: int = 10
    tool_timeout_s: float = 30.0
    tool_out_cap: int = 6000
    work_dir: str = "."
    exec_timeout_s: float = 30.0
    mcp_config: str | None = None
    use_tool_router: bool = True
    # paged LLM engine (GPU Qwen/MiniCPM only). Incoherent with the Gemma
    # CPU sidecar — forced off for backend="gemma" (see __init__).
    llm_paged: bool = False
    llm_paged_blocks: int = 16
    llm_paged_batch_size: int = 4
    # legacy text-queue knob (kept for API compat, unused by voice turns).
    max_queue: int = 16
    sandbox: object = None


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
    """VAD -> STT -> deep-agent LLM (Gemma CPU) -> TTS (GPU), streaming."""

    def __init__(self, config: VoiceAgentConfig | None = None):
        self.config = config or VoiceAgentConfig()
        cfg = self.config
        # Gemma sidecar owns its own inference (llama-server on CPU): the
        # paged GPU engine must never warm weights behind its back.
        llm_cfg = cfg.llm
        if str(getattr(llm_cfg, "backend", "")) == "gemma" and cfg.llm_paged:
            from dataclasses import replace

            llm_cfg = replace(llm_cfg, use_paged=False)
        elif cfg.llm_paged:
            from dataclasses import replace

            llm_cfg = replace(llm_cfg, use_paged=True,
                              paged_blocks=cfg.llm_paged_blocks,
                              paged_batch_size=cfg.llm_paged_batch_size)
        self.vad = VadModel(cfg.vad)
        self.stt = SttModel(cfg.stt)
        self.llm = LlmModel(llm_cfg)
        self.tts = TtsModel(cfg.tts)
        from src.agent.chat_model import LocalChatModel

        self.chat_model = LocalChatModel(llm=self.llm)
        try:
            from deepagents.backends import LocalShellBackend

            self.backend = LocalShellBackend(
                root_dir=cfg.work_dir,
                timeout=int(cfg.exec_timeout_s or cfg.tool_timeout_s))
        except Exception:
            self.backend = None
        try:
            from src.agent.mcp.client import load_extra_tools

            self._mcp_client, self._extra_tools = load_extra_tools(
                cfg.mcp_config)
        except Exception:
            self._mcp_client, self._extra_tools = None, []
        self.sessions = JsonSessionMemory(
            sessions_dir=cfg.sessions_dir,
            max_turns=cfg.max_session_turns,
            max_age_s=cfg.session_ttl_s,
            max_sessions=cfg.max_sessions,
        )
        try:
            from src.models.runtime import FIFOScheduler

            self._sched = FIFOScheduler(cfg.max_inflight)
        except Exception:
            self._sched = None
        self.missing: list = []
        self._warmed = False
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, list[str]] = {}
        self.approvals: dict[str, set[str]] = {}
        self._paged_engine = None
        self.agent = self._build_agent(self._extra_tools or [])

    # -- construction -------------------------------------------------
    def _build_agent(self, extra_tools):
        from deepagents import create_deep_agent
        from langchain.agents.middleware import TodoListMiddleware

        from src.agent.trim import TrimObservationsMiddleware

        middleware = [
            TodoListMiddleware(
                system_prompt="Plan multi-step work with write_todos; "
                              "skip it for single replies.",
                tool_description="Track multi-step work "
                                 "(one call per turn).",
            ),
            TrimObservationsMiddleware(limit=1500),
        ]
        if getattr(self.config, "use_tool_router", False):
            try:
                from src.agent.tool_router import InjectToolMiddleware

                middleware.append(InjectToolMiddleware())
            except Exception:
                pass
        try:
            if self.backend is not None:
                return create_deep_agent(
                    model=self.chat_model,
                    backend=self.backend,
                    tools=list(extra_tools or []),
                    middleware=middleware,
                    system_prompt=SYSTEM_PROMPT,
                )
        except Exception:
            pass
        from deepagents import create_deep_agent as _create

        return _create(
            model=self.chat_model,
            tools=list(extra_tools or []),
            middleware=middleware,
            system_prompt=SYSTEM_PROMPT,
        )

    @property
    def warmed(self) -> bool:
        return self._warmed

    async def warm(self) -> "VoiceAgent":
        """Build every leg; collect failures instead of raising."""
        self.missing = []
        for name in ("vad", "stt", "llm", "tts"):
            leg = getattr(self, name, None)
            if leg is None:
                continue
            try:
                await leg.warm()
            except Exception as exc:  # noqa: BLE001 - omit-and-report
                self.missing.append(f"{name} leg: {exc}")
        try:
            eng = None
            if hasattr(self.llm, "_paged_engine"):
                cand = getattr(self.llm, "_paged_engine")
                eng = cand() if callable(cand) else cand
                if eng is None and hasattr(self.llm, "_paged_engine_inst"):
                    eng = getattr(self.llm, "_paged_engine_inst")
            self._paged_engine = eng
        except Exception:
            self._paged_engine = None
        self._warmed = True
        return self

    @property
    def paged_engine(self):
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
        eng = self.paged_engine
        if eng is None:
            return None
        try:
            return eng.stats()  # type: ignore
        except Exception:
            return None

    # -- sessions (app-layer helpers delegate to the same store) -------
    def history(self, session_id: str | None) -> list:
        try:
            return self.sessions.history(session_id) if session_id else []
        except Exception:
            return []

    def export_session(self, session_id: str) -> list:
        return self.history(session_id)

    def import_session(self, session_id: str, data: list) -> None:
        try:
            from langchain_core.messages import AIMessage, HumanMessage

            msgs = []
            for m in data if isinstance(data, list) else []:
                if not isinstance(m, dict):
                    continue
                role, content = m.get("role"), str(m.get("content", "") or "")
                if role == "assistant":
                    msgs.append(AIMessage(content=content))
                elif role == "user":
                    msgs.append(HumanMessage(content=content))
            if session_id and msgs:
                for m in msgs:
                    kind = getattr(m, "type", "") or ""
                    role = "assistant" if kind == "ai" else "user"
                    self.sessions.append(str(session_id), role, m.content)
        except Exception:
            pass

    def remember_approval(self, session_id: str, command: str):
        if not session_id or not command:
            return
        self.approvals.setdefault(session_id, set()).add(
            command.strip().split()[0][:40])

    def is_approved(self, session_id: str | None, action) -> bool:
        if not session_id:
            return False
        allow = self.approvals.get(session_id, set())
        if not allow:
            return False
        cmd = getattr(action, "command", "") or ""
        first = cmd.strip().split()[0] if cmd.strip() else ""
        return first in allow or getattr(action, "op", "") in allow

    # -- admission ------------------------------------------------------
    async def _admit(self):
        if self._sched is None:
            return None
        ticket = await asyncio.to_thread(
            self._sched.acquire, True, self.config.queue_timeout_s)
        if ticket is None:
            raise TimeoutError(
                f"server saturated ({self._sched.max_concurrency} in flight, "
                f"queue waited {self.config.queue_timeout_s:.0f}s); retry")
        return ticket

    def _release(self, ticket) -> None:
        try:
            if ticket is not None and self._sched is not None:
                self._sched.release(ticket)
        except Exception:
            pass

    # -- shared turn machinery ------------------------------------------
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
        from src.tools.terminal import (
            parse_bare_tail,
            parse_terminal_action,
            parse_xml_action,
        )

        txt = str(content or "").strip()
        if not txt:
            return ""
        for cand in (parse_terminal_action(txt), parse_xml_action(txt),
                     parse_bare_tail(txt)):
            if cand is not None and cand.op == "done":
                return cand.reply or ""
        return txt

    def _confirm_wrapper(self, confirm_fn, sid: str | None):
        async def _wrapper(action):
            res = None
            if confirm_fn is not None:
                res = (await confirm_fn(action)
                       if asyncio.iscoroutinefunction(confirm_fn)
                       else confirm_fn(action))
            if isinstance(res, str) and res.lower() in ("always", "a",
                                                        "allowlist"):
                if sid:
                    self.remember_approval(
                        sid, getattr(action, "command", "") or
                        getattr(action, "op", ""))
                return "always"
            return bool(res) if res is not None else False

        return _wrapper

    async def _agent_invoke(self, text: str, *, sid: str | None,
                            cwd: str, confirm_fn, system_suffix: str,
                            out_turns: list):
        """One deep-agent invocation, yielding term/* mid-turn events."""
        from langchain_core.messages import AIMessageChunk, HumanMessage

        from src.models.llm import split_thinking
        from src.tools.terminal import is_degenerate

        confirm = self._confirm_wrapper(confirm_fn, sid)
        _ = confirm  # confirm gate lives in tool middleware for now
        try:
            # Rebind every turn: evals/tests hot-swap agent.llm/chat_model
            # after construction (llm-only warm); a graph compiled in
            # __init__ would otherwise keep calling the stale leg.
            agent = self._build_agent(self._extra_tools or [])
            self.agent = agent
        except Exception as exc:
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
                    msg, _meta = (payload if isinstance(payload, (tuple, list))
                                  else (payload, {}))
                    if isinstance(msg, AIMessageChunk):
                        for piece in _chunk_pieces(msg):
                            yield AgentEvent(node="term", kind="token",
                                             data={"piece": piece[:500]})
                    continue
                if not isinstance(payload, dict):
                    continue
                for _node, data in payload.items():
                    msgs = (data.get("messages", [])
                            if isinstance(data, dict) else [])
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
                                        data={"action": {"action": name,
                                                         **args}})
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
        except Exception as exc:  # never kill the turn on engine errors
            yield AgentEvent(node="term", kind="error",
                             data={"message": str(exc)[:300]})
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
        """Serialize turns per session; queue + inject contenders."""
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

    # -- text turns (TUI, bridge, CLI, evals) ----------------------------
    async def run_text(self, text: str, session_id: str | None = None,
                       cwd: str | None = None, confirm_fn=None):
        """Run one text turn, yielding term/audio/summary events."""
        from src.tools.terminal import is_degenerate

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
            if getattr(self, "tts", None) is not None:
                try:
                    out = await self.tts.speak(_speak_head(reply))
                    yield AgentEvent(node="term", kind="audio", data={
                        "wav": np.asarray(out.wav, dtype="float32"),
                        "sr": int(out.sample_rate), "sentence": reply[:500]})
                except Exception as exc:
                    yield AgentEvent(node="term", kind="error",
                                     data={"message": f"tts failed: {exc}"[:200]})
        yield AgentEvent(node="term", kind="summary", data={
            "text": str(text), "reply": last_reply,
            "thinking": (last_think or "")[:2000], "session_id": sid})

    # -- dual call: text in -> reply string; audio in -> event stream --
    def __call__(self, audio_or_text, sr: int = 16000,
                 session_id: str | None = None):
        """Text or voice dispatch (both transports share one object).

        - ``await agent("compute it", session_id="t")`` — plain text turn,
          returns the final reply string (CLI-style / unit tests).
        - ``async for ev in agent(wav, sr, sid)`` — voice turn, yields
          vad/stt/llm/tts/term events + summary (``server.py`` ``/talk``).
        """
        if isinstance(audio_or_text, str):
            return self._text_reply(str(audio_or_text),
                                    session_id=session_id)
        return self._voice_turns(audio_or_text, sr=sr,
                                 session_id=session_id)

    async def _text_reply(self, text: str,
                          session_id: str | None = None) -> str:
        """One text turn, reply string only (no audio synthesis).

        Goes through ``agent.ainvoke`` (the ``_agenerate_async`` path),
        mirroring the pre-voice text harness: session history in, final
        reply out, remembered as a turn.
        """
        from langchain_core.messages import HumanMessage

        self.agent = self._build_agent(self._extra_tools or [])
        hist = self.sessions.history(session_id) if session_id else []
        res = await self.agent.ainvoke(
            {"messages": [*_lc_messages(hist),
                          HumanMessage(content=str(text))]},
            config={"recursion_limit": self.config.recursion_limit},
        )
        msgs = res.get("messages", []) if isinstance(res, dict) else []
        reply = ""
        for m in reversed(list(msgs or [])):
            if isinstance(m, dict):
                role = m.get("type") or m.get("role") or ""
                content = m.get("content", "")
            else:
                role = getattr(m, "type", None) or ""
                content = getattr(m, "content", "")
            if role not in ("ai", "assistant"):
                continue
            if isinstance(content, list):
                content = " ".join(
                    str(c.get("text", c)) for c in content
                    if isinstance(c, dict))
            reply = str(content or "")
            break
        try:
            _thinking, answer = split_thinking(reply)
        except Exception:
            answer = reply
        reply = str(answer or reply or "").strip()
        if session_id and (text or reply):
            try:
                self.sessions.remember_turn(session_id, str(text), reply)
            except Exception:
                pass
        return reply

    # -- voice turns (server /talk) ---------------------------------------
    async def _voice_turns(self, audio, sr: int = 16000,
                           session_id: str | None = None):
        """Run one voice turn, yielding vad/stt/llm/tts/term events + summary."""
        from engine import SentenceSplitter

        try:
            from src.models.runtime import check_budget
        except Exception:
            check_budget = None
        wav = np.asarray(audio, dtype=np.float32).ravel()
        if wav.size == 0:
            raise ValueError("empty audio")
        dur_s = len(wav) / float(sr)
        if dur_s > self.config.max_audio_s:
            raise ValueError(
                f"audio {dur_s:.1f}s exceeds {self.config.max_audio_s:.0f}s cap")
        # VRAM guard covers GPU legs only (STT/TTS); the CPU LLM never trips it.
        if check_budget is not None:
            check_budget(self.config.per_turn_mb, self.config.vram_budget_mb,
                         what="voice turn")

        ticket = await self._admit()
        try:
            try:
                from engine import new_session_id
            except Exception:
                import uuid as _uuid

                def new_session_id():  # type: ignore
                    return _uuid.uuid4().hex
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
            audio_wav = (wav if not segments
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
                res = await self.stt.transcribe(audio_wav, int(sr))
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
                        "piece": str(ev.data.get("piece", "") or ""),
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

                async def _feed(sentences, speak):
                    for sent in sentences:
                        if (sent or "").strip():
                            async for ev in speak(sent):
                                yield ev

                # feed whole reply through the splitter, then flush
                for sent in splitter.push(reply):
                    pending.append(sent)
                tail = splitter.flush()
                if tail:
                    pending.append(tail)
                async for audio_ev in _feed(
                        [s for s in pending if (s or "").strip()],
                        _speak_sentence):
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
            self._release(ticket)
