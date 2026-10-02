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
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np

from src.agent.delegate import Route, is_task_shaped
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
# Front-brain (two-brain mode) CHAT system prompt. Used only on the branch
# where the boolean router said "this is conversation". Spoken aloud, so it
# is short; the escalation question itself lives in
# src/agent/delegate.py (BOOLEAN_SYSTEM), and the front leg is never given
# tool specs — a 0.6B holding a tool schema answers the tool, not the user.
_FRONT_SYSTEM = (
    "You are a voice assistant. The user is talking, not asking you to do "
    "anything on their computer. Reply in ONE short plain sentence, under 15 "
    "words. No tools, no preamble, no lists.")
_VOICE_FAST_SYSTEM = ("You are a voice assistant. Reply in one very short "
                      "sentence, under 10 words. Plain text only — no tools, "
                      "no preamble, just the reply.")
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
    # paged LLM engine (GPU Qwen/MiniCPM only). Incoherent with sidecar
    # backends — forced off for backend="gemma"/"bonsai" (see __init__).
    llm_paged: bool = False
    llm_paged_blocks: int = 16
    llm_paged_batch_size: int = 4
    # Text turns speak their reply (TTS alloc on GPU) by default. On 4GB
    # cards next to a 27B LLM there is no headroom left — set False for
    # text-only operation (TUI auto mode): no audio event, no VRAM touch,
    # no red OOM blob next to a perfectly good reply. Voice turns and
    # /term/say are unaffected. Bridge sets it from VOICE_TEXT_ONLY=1.
    speak_text_turns: bool = True
    # Fast voice turns: VAD -> STT -> ONE direct LLM generate -> TTS,
    # bypassing the deep-agent loop (graph rebuild + tool router + up to 6
    # steps). Same four legs attached, same events out. The full agent
    # path stays for tool-using turns; fast is the latency play (~400ms
    # E2E budget) for conversational voice.
    fast_voice: bool = False
    # legacy text-queue knob (kept for API compat, unused by voice turns).
    max_queue: int = 16
    sandbox: object = None
    # Three-tier memory (all off by default — prod behavior unchanged).
    # memory_tokens: L1 token-budget window (0 = count cap only).
    # memory_recall: inject L2 episodes + L3 facts into the turn prompt.
    # memory_store: consolidate evicted L1 turns into L2 on remember.
    # memory_dir: local state root (episodic.db, facts.md — gitignored).
    memory_tokens: int = 0
    memory_recall: bool = False
    memory_store: bool = False
    memory_dir: str = "memory"
    # Two-brain delegation (src/agent/delegate.py). On: a small front model
    # (Qwen3-0.6B) talks to the user and is given no tools — it only answers
    # yes/no on whether the turn needs the worker, and an escalated turn runs
    # on the worker leg (Bonsai 27B) with the whole agentic harness. Off: the
    # single llm leg answers everything, as before.
    #
    # Default OFF on purpose: turning it on builds a SECOND set of weights,
    # so a library caller that never asked for delegation should not wake up
    # paying for it (VRAM on a 4GB card, plus a warm). The app entry points
    # (server.py /talk, bridge.py /term) opt in — that is where the
    # two-brain design is the product decision, not a library detail.
    delegate: bool = False
    delegate_config: str | None = None
    # Worker sidecar warm policy. False (default): the Bonsai worker is NOT
    # warmed at boot — zero CPU/GPU footprint until the first delegate
    # route boots it on demand (see _ensure_worker). True: warm eagerly as
    # before. `configs/delegate.yaml` may also set `worker_eager: true`.
    worker_eager: bool = False


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
    """VAD -> STT -> deep-agent LLM (Bonsai 27B sidecar) -> TTS (GPU), streaming."""

    def __init__(self, config: VoiceAgentConfig | None = None):
        self.config = config or VoiceAgentConfig()
        cfg = self.config
        # Sidecar legs own their own inference (llama-server on CPU, or
        # Bonsai with partial GPU offload): the paged GPU engine must
        # never warm weights behind their back.
        llm_cfg = cfg.llm
        from src.models.llm import SIDECAR_BACKENDS

        # -- two-brain routing -------------------------------------------
        # Loaded before the legs so `worker:` overrides in the YAML land on
        # the single existing llm leg. Delegation never forks the worker
        # into a second instance: same graph, same tools, same memory.
        self.delegate_cfg = None
        self.front_llm = None
        # Router-vs-regex disagreement counters, surfaced on /health. See
        # _note_route_disagreement.
        self._route_skew: dict = {}
        if bool(getattr(cfg, "delegate", False)):
            try:
                from src.agent.delegate import (
                    front_llm_config, load_config, worker_llm_overrides)

                dcfg = load_config(getattr(cfg, "delegate_config", None))
                if dcfg.enabled:
                    self.delegate_cfg = dcfg
                    over = worker_llm_overrides(dcfg)
                    if over:
                        llm_cfg = replace(llm_cfg, **over)
            except Exception:
                self.delegate_cfg = None
        if str(getattr(llm_cfg, "backend", "")) in SIDECAR_BACKENDS and cfg.llm_paged:
            llm_cfg = replace(llm_cfg, use_paged=False)
        elif cfg.llm_paged:
            llm_cfg = replace(llm_cfg, use_paged=True,
                              paged_blocks=cfg.llm_paged_blocks,
                              paged_batch_size=cfg.llm_paged_batch_size)
        self.vad = VadModel(cfg.vad)
        self.stt = SttModel(cfg.stt)
        self.llm = LlmModel(llm_cfg)
        self.tts = TtsModel(cfg.tts)
        # Front leg: a SECOND LlmModel, independent of the worker. Built
        # after the worker override so a placement failure can never take
        # the worker down with it.
        if self.delegate_cfg is not None:
            try:
                from src.agent.delegate import front_llm_config

                self.front_llm = LlmModel(front_llm_config(self.delegate_cfg))
            except Exception:
                self.front_llm = None
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
            max_tokens=getattr(cfg, "memory_tokens", 0) or 0,
        )
        # Shared semantic tool router: one bge-small load per process, not
        # one per turn (_build_agent used to construct a fresh router each
        # turn, re-reading 199 weight tensors from disk every time).
        self._router = None
        self._episodic = None  # lazy EpisodicStore (memory_store/recall)
        try:
            from src.models.runtime import FIFOScheduler

            self._sched = FIFOScheduler(cfg.max_inflight)
        except Exception:
            self._sched = None
        self.missing: list = []
        self._warmed = False
        # Lazy-worker state (see _ensure_worker): the Bonsai sidecar boots
        # on first delegate route, not at app warm. A refusal latches so
        # later YES turns fail fast with the same reason; an explicit
        # /model switch to a warmed leg resets both (bridge.py).
        self._worker_unavailable: str | None = None
        self._worker_warmed = False
        self._worker_eager = bool(getattr(cfg, "worker_eager", False))
        if self.delegate_cfg is not None and bool(
                getattr(self.delegate_cfg, "worker_eager", False)):
            # YAML asks eager too: either source opting in enables it.
            self._worker_eager = True
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, list[str]] = {}
        self.approvals: dict[str, set[str]] = {}
        self._paged_engine = None
        self.agent = self._build_agent(self._extra_tools or [])

    # -- construction -------------------------------------------------
    def _summarization_middleware(self, backend):
        """In-turn compaction guard (returns None when unavailable).

        deepagents' SummarizationMiddleware with OUR estimator and a
        tool-stripped summarizer sharing the warmed leg (no extra warm,
        thinking capped by the leg config). Scope is strictly in-turn
        growth: trigger near ctx, keep recent steps. Cross-turn memory
        stays in sessions/L2/L3 — this middleware is amnesiac across
        turns (the graph is rebuilt per turn), so it must never be the
        only compressor. File offload stays OFF for the same reason
        (two writers, one summary).
        """
        try:
            from deepagents.middleware.summarization import (
                SummarizationMiddleware)

            from src.agent.budget import CONTEXT_TOKENS, count_messages
            from src.agent.chat_model import LocalChatModel

            backend_id = str(getattr(
                getattr(self.llm, "config", None), "backend", ""))
            ctx = int(CONTEXT_TOKENS.get(backend_id, 8192))
            summarizer = LocalChatModel(llm=self.llm, no_tools=True)
            return SummarizationMiddleware(
                model=summarizer,
                backend=backend,
                trigger=("tokens", max(2048, int(ctx * 0.9))),
                keep=("messages", 6),
                token_counter=count_messages,
                trim_tokens_to_summarize=1500,
            )
        except Exception:
            return None

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

                if self._router is None:
                    from src.agent.tool_router import SemanticRouter

                    self._router = SemanticRouter()
                middleware.append(InjectToolMiddleware(router=self._router))
            except Exception:
                pass
        if self.backend is not None:
            try:
                summ = self._summarization_middleware(self.backend)
                if summ is not None:
                    middleware.append(summ)
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
        # Order matters. The front leg is small and must be resident BEFORE
        # the worker warms: the Bonsai sidecar picks its layer split with
        # `-ngl auto`, which probes *free* VRAM at startup. Warm the worker
        # first and it claims the whole 4GB card, the front then fails to
        # allocate, and every turn degrades to the worker. Small-and-fixed
        # first, adaptive-and-large second.
        for name in ("vad", "stt", "tts", "front_llm", "llm"):
            if name == "tts" and not self.config.speak_text_turns:
                self.missing.append("tts leg: skipped (text-only)")
                continue
            if (name == "llm" and self.delegate_cfg is not None
                    and not self._worker_eager):
                # Lazy worker: no footprint until the first delegate route
                # boots it on demand. Deliberately NOT a missing entry —
                # nothing failed.
                continue
            leg = getattr(self, name, None)
            if leg is None:
                continue
            try:
                await leg.warm()
                if name == "llm":
                    self._worker_warmed = True
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

    # -- two-brain routing -------------------------------------------------
    def _front_messages(self, text: str, history: list) -> list:
        """Front-model CHAT prompt: one short spoken sentence, no tools.

        Deliberately NOT ``messages_for_terminal``: the front model has no
        terminal, and handing it the op catalogue is how small models start
        emitting ops they cannot run. No tool specs at all, for the same
        reason.
        """
        msgs = [{"role": "system", "content": _FRONT_SYSTEM}]
        msgs.extend(history or [])
        msgs.append({"role": "user", "content": str(text)[:1500]})
        return msgs

    async def _front_decide(self, text: str, history: list):
        """Route one turn: does this need the worker, or just a reply?

        Returns a :class:`~src.agent.delegate.Route`. The front leg gets no
        tool schema and writes no task string — it answers one yes/no
        question, and the worker re-reads the raw user text itself. Every
        failure path escalates: a dead front leg, an unparseable answer, or a
        narration-shaped reply all route to the worker, because that is the
        leg that can actually do anything.

        Order matters: the boolean is asked FIRST so a delegated turn costs
        one 6-token call and never generates a chat reply nobody will hear.
        Asking for the flag and the reply together was measured and loses
        real tasks (31/37 vs 34/37, six tasks silently dropped).
        """
        from src.agent.delegate import decide_escalate, parse_delegate

        cap = int(getattr(self.delegate_cfg, "max_task_chars", 1200) or 1200)
        escalate, reason = await decide_escalate(self.front_llm, text)
        if escalate:
            return Route("delegate", str(text), reason=reason,
                         forced=reason != "boolean")

        # Front said this is conversation, so it is now free to speak.
        reply, reason2 = await self._front_chat(text, history)
        route = parse_delegate(reply, max_task_chars=cap)
        if route is not None:
            return route
        # Backstop: the front model said "chat" but the request reads like
        # machine work. Cheap, deterministic, and the last guard against
        # telling the user something is done that never happened.
        if (getattr(self.delegate_cfg, "backstop", False)
                and is_task_shaped(text, int(getattr(
                    self.delegate_cfg, "backstop_min_score", 2) or 2))):
            return Route("delegate", str(text), reason="backstop", forced=True)
        return Route("chat", reply, reason=reason2 or reason)

    async def _front_chat(self, text: str, history: list) -> tuple[str, str]:
        """One short spoken reply from the front leg.

        Returns ``(reply, reason)``. Never raises: a dead front leg on the
        chat path is an empty reply, and an empty chat reply falls through
        to the worker in _run_turn rather than answering with silence.
        """
        try:
            res = await self.front_llm.generate(
                self._front_messages(text, history), 96)
            return str(getattr(res, "text", "") or "").strip(), "chat"
        except Exception as exc:  # noqa: BLE001 - fall through to the worker
            return "", f"chat-error:{type(exc).__name__}"

    async def _ensure_worker(self) -> str:
        """Warm the lazy worker leg on first delegate route.

        Returns "" when the worker is usable, else the refusal reason.
        A refusal latches in _worker_unavailable so later YES turns fail
        fast with the same message instead of re-trying a doomed warm.
        An explicit /model switch to a warmed leg resets the latch (see
        bridge.py model switch).
        """
        if self.delegate_cfg is None:
            return ""
        if getattr(self, "_worker_unavailable", None):
            return str(self._worker_unavailable)
        if getattr(self, "_worker_warmed", False):
            return ""
        leg = getattr(self, "llm", None)
        if leg is None:
            reason = "no worker leg constructed"
            self._worker_unavailable = reason
            return reason
        try:
            await leg.warm()
        except Exception as exc:  # noqa: BLE001 - latch, report, move on
            reason = str(exc)[:200] or type(exc).__name__
            self._worker_unavailable = reason
            try:
                self.missing.append(f"llm leg: {reason}")
            except Exception:
                pass
            return reason
        self._worker_warmed = True
        return ""

    @staticmethod
    def _worker_down_notice(task: str, reason: str) -> tuple:
        """Transcript entry for a delegated turn the worker could not run."""
        return (str(task), (
            "I handed that to the worker agent, but it could not "
            f"start, so nothing was done: {reason}"), "")

    async def _agent_invoke(self, text: str, *, sid: str | None,
                            cwd: str, confirm_fn, system_suffix: str,
                            out_turns: list):
        """One deep-agent invocation, yielding term/* mid-turn events.

        With delegation on, this is the WORKER leg: it only runs for turns
        the front brain handed over (or forced over via the backstop), and
        it runs the full existing graph.
        """
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
        mem_ctx = self._memory_context(sid, str(text)) if sid else ""
        lc_msgs = _lc_messages(history) + [
            HumanMessage(content=str(text) + mem_ctx +
                         f"\nCWD: {cwd} SHELL: bash")]
        final_reply = ""
        thinking = ""
        last_think = ""
        # True once live think deltas streamed this step: the updates-mode
        # copy of the same trace must not re-render (it would duplicate the
        # grown bubble — and in a different, newline-joined format).
        # Display-only flag: `thinking`/`last_think` still update so the
        # summary, memory and TTS gating keep correct data.
        live_think_shown = False
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
                        # Live think lane: reasoning deltas stream here while
                        # the step is still generating (the full trace lands
                        # via the updates-mode thinking event at step end).
                        ak = getattr(msg, "additional_kwargs", None) or {}
                        if isinstance(ak, dict) and ak.get("thinking_delta"):
                            live_think_shown = True
                            yield AgentEvent(
                                node="term", kind="thinking",
                                data={"text": str(ak["thinking_delta"])[:500],
                                      "append": True})
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
                                if live_think_shown:
                                    live_think_shown = False
                                else:
                                    yield AgentEvent(
                                        node="term", kind="thinking",
                                        data={"text": think[:2000]})
                            if tool_calls:
                                live_think_shown = False  # new step
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
                                live_think_shown = False
                                final_reply = self._resolve_reply(content)
                        elif role == "tool":
                            live_think_shown = False
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
            if not live_think_shown:
                yield AgentEvent(node="term", kind="thinking",
                                 data={"text": thinking[:2000]})
            live_think_shown = False
        if not reply and did_work:
            reply = "Done."
        if reply and is_degenerate(reply):
            yield AgentEvent(node="term", kind="error",
                             data={"message": "model output degenerate"})
            reply = "Sorry — I garbled that. Try rephrasing."
        if sid and (text or reply):
            try:
                before = [m.get("content", "") for m in
                          self.sessions.history(sid)] if getattr(
                              self.config, "memory_store", False) else []
                self.sessions.remember_turn(sid, str(text), reply)
                if before:
                    self._memory_consolidate(sid, before)
            except Exception:
                pass
        out_turns.append((str(text), reply, thinking))

    async def _run_locked(self, text: str, *, sid: str | None, cwd: str,
                          confirm_fn, system_suffix: str, out_turns: list):
        """Serialize turns per session; queue + inject contenders.

        The lock is taken BEFORE the front brain is consulted, not just
        around the worker: two concurrent turns on one session would
        otherwise both ask the front model about a conversation it has not
        seen the first half of, and the second could answer from stale
        context. The front call is also the slow part on a CPU placement,
        so serializing it keeps the session coherent.
        """
        lock = None
        if sid is not None:
            lock = self._locks.setdefault(sid, asyncio.Lock())
            if lock.locked():
                pend = self._pending.setdefault(sid, [])
                pend.append(str(text))
                yield AgentEvent(node="term", kind="queued",
                                 data={"position": len(pend), "session_id": sid})
                return
            await lock.acquire()
        try:
            async def _one(t: str):
                async for ev in self._run_turn(
                        t, sid=sid, cwd=cwd, confirm_fn=confirm_fn,
                        system_suffix=system_suffix, out_turns=out_turns):
                    yield ev

            if sid is None:
                async for ev in _one(str(text)):
                    yield ev
                return
            texts = self._pending.pop(sid, []) + [str(text)]
            while texts:
                async for ev in _one(texts.pop(0)):
                    yield ev
                texts.extend(self._pending.pop(sid, []))
        finally:
            if lock is not None:
                lock.release()

    async def _run_turn(self, text: str, *, sid: str | None, cwd: str,
                        confirm_fn, system_suffix: str, out_turns: list):
        """Route one turn: front brain decides chat vs delegate.

        A chat route is answered here and never touches the worker graph,
        which is what keeps small talk off the 27B. A delegate route (or the
        backstop forcing one) runs the worker agent and its result is what
        the user hears — the front model does not get to reword it, so a
        correct result can't be garbled on its way out.
        """
        if self.front_llm is None or self.delegate_cfg is None:
            reason = await self._ensure_worker()
            if reason:
                out_turns.append(
                    self._worker_down_notice(str(text), reason))
                return
            async for ev in self._agent_invoke(
                    text, sid=sid, cwd=cwd, confirm_fn=confirm_fn,
                    system_suffix=system_suffix, out_turns=out_turns):
                yield ev
            return

        history = self.sessions.history(sid) if sid else []
        route = await self._front_decide(text, history)
        self._note_route_disagreement(text, route)

        if getattr(self.delegate_cfg, "emit_route", True):
            yield AgentEvent(node="term", kind="route", data={
                "brain": "front" if route.kind == "chat" else "worker",
                "kind": route.kind, "reason": str(route.reason or "")[:80],
                "forced": bool(route.forced),
            })

        if route.kind == "chat":
            reply = str(route.text or "").strip()
            if not reply:
                # Front model returned nothing. Falling through to the
                # worker beats answering with silence.
                reason = await self._ensure_worker()
                if reason:
                    out_turns.append(
                        self._worker_down_notice(str(text), reason))
                    return
                async for ev in self._agent_invoke(
                        text, sid=sid, cwd=cwd, confirm_fn=confirm_fn,
                        system_suffix=system_suffix, out_turns=out_turns):
                    yield ev
                return
            out_turns.append((str(text), reply, ""))
            return

        task = str(route.text or "").strip() or str(text)
        reason = await self._ensure_worker()
        if reason:
            out_turns.append(self._worker_down_notice(task, reason))
            return
        before = len(out_turns)
        async for ev in self._agent_invoke(
                task, sid=sid, cwd=cwd, confirm_fn=confirm_fn,
                system_suffix=system_suffix, out_turns=out_turns):
            yield ev
        # "No turn appended" is not the only empty case: a failed worker
        # still records a turn, just with an empty reply, and run_text's
        # post-loop then clobbers last_reply with "" and emits no chat
        # event. So test the last reply, not the turn count.
        produced = any(str(reply).strip()
                       for (_t, reply, _th) in out_turns[before:])
        if not produced:
            # Delegated, but nothing came back. If the worker leg failed to
            # warm, say THAT instead of returning nothing: the front model
            # already took the request, so silence here reads as "done" and
            # is the one outcome worse than a clear failure.
            why = self._worker_missing_reason()
            if why:
                notice = (str(text), (
                    "I handed that to the worker agent, but it could not "
                    f"start, so nothing was done: {why}"), "")
                if len(out_turns) > before:
                    # Replace the empty placeholder rather than adding a
                    # second entry for the same turn.
                    out_turns[-1] = notice
                else:
                    out_turns.append(notice)

    def _note_route_disagreement(self, text: str, route: Route) -> None:
        """Log where the 0.6B and the regex scorer disagree.

        The boolean is the router; :func:`is_task_shaped` no longer decides
        anything except the backstop override. But the two were measured on
        opposite failure axes — the boolean is reliable on real tasks and
        over-eager on chit-chat, the regex is the reverse — so the cases
        where they disagree are the interesting ones and the only honest way
        to tune the 0.6B on live traffic rather than on 37 prompts.
        """
        try:
            scored = is_task_shaped(text, int(getattr(
                self.delegate_cfg, "backstop_min_score", 2) or 2))
            said = route.kind == "delegate"
            if scored == said:
                return
            self._route_skew.setdefault(
                "regex_no_model_yes" if said else "model_no_regex_yes", 0)
            self._route_skew[
                "regex_no_model_yes" if said else "model_no_regex_yes"] += 1
            log.info("route skew: model=%s regex=%s %.90s", route.kind, scored,
                     str(text))
        except Exception:  # noqa: BLE001 - telemetry must never break a turn
            pass

    def _worker_missing_reason(self) -> str:
        """Why the worker leg is unusable this process, or "" if it is fine.

        ``warm()`` collects leg failures instead of raising, so a dead
        worker is otherwise invisible until a delegated turn answers with
        nothing at all. Also covers the lazy-worker latch: a refused
        on-demand warm reports the same way as a failed boot warm.
        """
        latched = getattr(self, "_worker_unavailable", None)
        if latched:
            return str(latched)[:200]
        for entry in list(getattr(self, "missing", None) or []):
            text = str(entry)
            if text.startswith("llm leg:"):
                return text.split(":", 1)[1].strip()[:200]
        return ""

    # -- three-tier memory ------------------------------------------------
    def _episodic_store(self):
        """Lazy L2 store (None unless memory_store/recall enabled)."""
        if self._episodic is None and (
                getattr(self.config, "memory_store", False)
                or getattr(self.config, "memory_recall", False)):
            try:
                from src.agent.episodic import EpisodicStore

                self._episodic = EpisodicStore(
                    os.path.join(getattr(self.config, "memory_dir",
                                         "memory"), "episodic.db"))
            except Exception:
                self._episodic = None
        return self._episodic

    def _memory_consolidate(self, sid, before_contents) -> None:
        """Store L1-evicted turns as L2 episodes (extractive summary)."""
        try:
            after = {m.get("content", "") for m in
                     self.sessions.history(sid)}
            dropped = [c for c in (before_contents or [])
                       if c and c not in after]
            if not dropped:
                return
            from src.agent.episodic import summarize_turns

            store = self._episodic_store()
            if store is None:
                return
            summary = summarize_turns(
                [{"role": "user", "content": c} for c in dropped])
            if summary:
                store.store(summary, session=str(sid or ""))
        except Exception:
            pass

    def _memory_context(self, sid, text: str) -> str:
        """L2 episodes + L3 facts injected above the CWD trailer."""
        if not getattr(self.config, "memory_recall", False):
            return ""
        try:
            parts = []
            store = self._episodic_store()
            if store is not None:
                rows = store.recall(text, k=3, session=str(sid or ""))
                if rows:
                    parts.append(
                        "\n[Past episodes — USE these to answer when "
                        "relevant; prefer them over guessing]\n" + "\n".join(
                            "- " + r["summary"][:300] for r in rows))
            from src.agent.facts import format_block, load_facts

            facts = load_facts(os.path.join(
                getattr(self.config, "memory_dir", "memory"), "facts.md"))
            if facts:
                parts.append("\n[User facts]\n" + format_block(facts))
            return "".join(parts)
        except Exception:
            return ""

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
            if getattr(self.config, "speak_text_turns", True) and getattr(
                    self, "tts", None) is not None:
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
        # fast_voice is the one-shot voice path and is incompatible with
        # delegation: it calls llm.generate directly, so a task would be
        # answered by whichever leg happens to be the worker with no tools
        # in front of it. Delegation wins — the front brain still keeps
        # chat turns cheap, just via the router instead of the short path.
        if (bool(getattr(self.config, "fast_voice", False))
                and self.delegate_cfg is None):
            return self._voice_fast(audio_or_text, sr=sr,
                                    session_id=session_id)
        return self._voice_turns(audio_or_text, sr=sr,
                                 session_id=session_id)

    async def _text_reply(self, text: str,
                          session_id: str | None = None) -> str:
        """One text turn, reply string only (no audio synthesis).

        With delegation on, the front brain routes first: a chat route
        answers here and never builds the worker graph at all, which is the
        whole point of the split on a text path that otherwise pays for
        agent setup on "hi".
        """
        from langchain_core.messages import HumanMessage

        hist = self.sessions.history(session_id) if session_id else []
        if self.front_llm is not None and self.delegate_cfg is not None:
            route = await self._front_decide(str(text), hist)
            if getattr(self.delegate_cfg, "emit_route", True):
                # No event stream on this path; keep the decision visible
                # on stdout for the CLI/eval callers that read it.
                print(f"[route] {'front' if route.kind == 'chat' else 'worker'}"
                      f" ({route.reason})", flush=True)
            if route.kind == "chat" and str(route.text or "").strip():
                reply = str(route.text).strip()
                if session_id:
                    try:
                        self.sessions.remember_turn(
                            session_id, str(text), reply)
                    except Exception:
                        pass
                return reply
            task = str(route.text or "").strip() or str(text)
        else:
            task = str(text)

        reason = await self._ensure_worker()
        if reason:
            return self._worker_down_notice(task, reason)[1]

        self.agent = self._build_agent(self._extra_tools or [])
        res = await self.agent.ainvoke(
            {"messages": [*_lc_messages(hist),
                          HumanMessage(content=task)]},
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
    async def _voice_fast(self, audio, sr: int = 16000,
                          session_id: str | None = None):
        """Latency fast path: VAD -> STT -> ONE llm.generate -> TTS.

        Same four legs, same event shapes as :meth:`_voice_turns`, but no
        deep-agent loop: no graph rebuild, no tool router, no multi-step
        iterations. Prompt is a small system line + capped history + the
        utterance, so prefill stays tiny and the reply ends at EOS within
        a few dozen tokens.
        """
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
        if check_budget is not None:
            check_budget(self.config.per_turn_mb, self.config.vram_budget_mb,
                         what="voice turn")

        from src.models.runtime.device import keepalive_start
        ticket = await self._admit()
        stop_keep = lambda: None  # noqa: E731 - rebound below; finally-safe
        try:
            try:
                from engine import new_session_id
            except Exception:
                import uuid as _uuid

                def new_session_id():  # type: ignore
                    return _uuid.uuid4().hex
            sid = session_id or new_session_id()
            remember = bool(session_id)
            eff_sid = sid if remember else None
            t0 = time.perf_counter()
            node_s: dict = {}
            stop_keep = keepalive_start()
            # Rev CPU+GPU clocks once per turn: after idle both sit at
            # minimum frequency and every phase re-pays ramp (~50-100 ms
            # each). One ~5 ms burst up front + the keepalive thread holds
            # P-state for the rest of the turn.
            try:
                from src.models.runtime.device import gpu_rev

                node_s["rev"] = float(gpu_rev())
                _rev_a = np.matmul(np.zeros((128, 128), dtype=np.float32),
                                   np.zeros((128, 128), dtype=np.float32))
            except Exception:
                pass

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

            # -- LLM: one direct generate (no agent loop) -----------------
            t_llm = time.perf_counter()
            try:
                hist = []
                if eff_sid:
                    try:
                        for m in self.sessions.history(eff_sid)[-2:]:
                            hist.append({
                                "role": str(m.get("role", "user")),
                                "content": str(m.get("content", ""))[:200],
                            })
                    except Exception:
                        hist = []
                msgs = ([{"role": "system", "content": _VOICE_FAST_SYSTEM}]
                        + hist
                        + [{"role": "user", "content": text}])
                gen = await self.llm.generate(msgs)
                thinking, reply = split_thinking(str(gen.text or ""))
                reply = str(reply or "").strip()
            finally:
                node_s["llm"] = node_s.get("llm", 0.0) + time.perf_counter() - t_llm
            if reply:
                try:
                    n_ids = len(getattr(gen, "output_ids", ()) or ())
                except Exception:
                    n_ids = 0
                yield AgentEvent(node="llm", kind="done",
                                 data={"text": reply, "ids": n_ids})
            if eff_sid and (text or reply):
                try:
                    self.sessions.remember_turn(eff_sid, str(text), reply)
                except Exception:
                    pass

            # -- TTS: single speak of the short reply ---------------------
            first_audio_at = None
            if (reply or "").strip():
                t_tts = time.perf_counter()
                try:
                    out = await self.tts.speak(reply)
                finally:
                    node_s["tts"] = (node_s.get("tts", 0.0)
                                     + time.perf_counter() - t_tts)
                first_audio_at = time.perf_counter() - t0
                yield AgentEvent(node="tts", kind="audio", data={
                    "wav": np.asarray(out.wav, dtype=np.float32),
                    "sr": int(out.sample_rate),
                    "sentence": str(out.sentence or reply),
                    "synth_s": float(out.synth_s),
                })

            total = time.perf_counter() - t0
            yield AgentEvent(node="turn", kind="summary", data={
                "text": text, "reply": reply, "segments": segments,
                "node_s": dict(node_s),
                "ttfa_s": first_audio_at if first_audio_at is not None else total,
                "total_s": total, "vram_mb": _vram_mb(), "session_id": sid})
        finally:
            try:
                stop_keep()
            except Exception:
                pass
            self._release(ticket)

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
                                 "error", "queued", "route"):
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
