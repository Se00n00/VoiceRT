"""Long-running agent: sessions, turns, tools, box.

Owns construction, graph assembly and the text entry; voice legs live on
``src/cascade.py``; two-brain routing composes on ``src/main.py``.
"""

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.agent.chat_model import LocalChatModel
from src.agent.delegate import (
    Route,
    decide_escalate,
    front_llm_config,
    is_task_shaped,
    load_config,
    parse_delegate,
    worker_llm_overrides,
)
from src.agent.episodic import EpisodicStore, summarize_turns
from src.agent.events import AgentEvent
from src.agent.facts import format_block, load_facts
from src.agent.mcp.client import load_extra_tools
from src.agent.memory import JsonSessionMemory
from src.agent.middleware import (
    ErrorHintMiddleware,
    InjectToolMiddleware,
    PolicyMiddleware,
    SummarizationMiddleware,
    TrimObservationsMiddleware,
    VerifyMiddleware,
)
from src.models.llm import (
    SIDECAR_BACKENDS,
    LlmConfig,
    LlmModel,
    split_thinking,
)
from src.models.runtime import FIFOScheduler, check_budget, max_allocated_mb
from src.prompts.base import SYSTEM_PROMPT, WORKER_STANCE
from src.prompts.memory import PAST_EPISODES_HEADER, USER_FACTS_HEADER
from src.prompts.middleware import TODO_SYSTEM_PROMPT, TODO_TOOL_DESCRIPTION
from src.prompts.voice import (
    DEGENERATE_REPLY,
    FRONT_CHAT_MAX_TOKENS,
    FRONT_SYSTEM,
    TEXT_SUFFIX,
    WORKER_DOWN_TEMPLATE,
    identity_sentence,
)

from src.agent.chat_model import (
    is_degenerate,
    parse_bare_tail,
    parse_terminal_action,
    parse_xml_action,
)

__all__ = ["LongRunningConfig", "LongRunningAgent"]

log = logging.getLogger(__name__)
"""Telemetry logger (route skew, never breaks turns)."""

class LongRunningConfig(BaseModel):
    """Agentic loop config. One class, one job.

    ..
    """

    model_config = ConfigDict(frozen=True)

    llm: LlmConfig = Field(default_factory=LlmConfig)
    work_dir: str = "."
    exec_timeout_s: float = 30.0
    tool_timeout_s: float = 30.0
    mcp_config: str | None = None
    use_tool_router: bool = False
    tool_top_k: int = 5
    max_agent_steps: int = 6
    recursion_limit: int = 10
    speak_text_turns: bool = True
    max_inflight: int = 4
    queue_timeout_s: float = 10.0
    max_session_turns: int = 8
    session_ttl_s: float = 1800.0
    max_sessions: int = 1000
    sessions_dir: str = "~/.voicert"
    memory_tokens: int = 0
    memory_recall: bool = False
    memory_store: bool = False
    memory_dir: str = "memory"
    delegate: bool = False
    delegate_config: str | None = None
    worker_eager: bool = False
    sandbox: Any = None


MAX_SPOKEN_WORDS = 500


_DELEGATION_TALK = re.compile(
    r"delegat|escalat|larger model|bigger model|hand(?:ing|ed|le)? "
    r"(it|this|that|things|off)|pass(?:ing|ed)? (it|this|that) (on|off|up)|"
    r"i(?:'| wi)ll (let|have|get) (someone|another|the) ",
    re.IGNORECASE,
)


class _BoxError(RuntimeError):
    """Turn-box setup failure (rejected cwd, docker down). Never silent."""


_BOX_DENY_PREFIXES = (
    "/proc",
    "/sys",
    "/dev",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/boot",
    "/root",
    "/var",
)
"""Host prefixes a turn cwd may never mount into a container."""


def _jail_turn_cwd(cwd: str, sessions_dir: str) -> str | None:
    """Requested cwd -> jailed host abspath, None when rejected.

    Rejects filesystem root, system prefixes, the session store, and
    home secret dirs; creates anything else. Pure except the mkdir.
    ..
    """
    raw = os.path.abspath(os.path.expanduser(str(cwd or ".") or "."))
    if raw == os.sep:
        return None
    guarded = _BOX_DENY_PREFIXES + (
        os.path.abspath(os.path.expanduser(str(sessions_dir or ""))),
    )
    for prefix in guarded:
        if not prefix or prefix == os.sep:
            continue
        if raw == prefix or raw.startswith(prefix + os.sep):
            return None
    home = os.path.expanduser("~")
    for secret in (".ssh", ".gnupg"):
        g = os.path.join(home, secret)
        if raw == g or raw.startswith(g + os.sep):
            return None
    try:
        os.makedirs(raw, exist_ok=True)
    except Exception:
        return None
    return raw


def _cap_words(text: str, limit: int = MAX_SPOKEN_WORDS) -> str:
    """Hard word cap for spoken replies (model instruction is advisory)."""
    words = str(text or "").split()
    if len(words) <= limit:
        return str(text or "")
    return " ".join(words[:limit])


def _speak_head(full: str, limit: int = 280) -> str:
    """Short speakable head of a reply (full text stays on screen)."""
    t = " ".join(str(full or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    dot = cut.rfind(". ")
    head = cut[: dot + 1] if dot > 120 else cut
    return head.strip() + " … full output is on screen."


def _lc_messages(history: list) -> list:
    """Session dicts -> LangChain messages (user/assistant only)."""
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
    """Extract non-empty text pieces from a streamed model chunk."""
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


def _vram_mb():
    """Current process GPU allocation in MB, or None without CUDA."""
    try:
        return float(max_allocated_mb())
    except Exception:
        return None


class LongRunningAgent:
    """Agentic loop: sessions, turns, tools, box. Owns __init__/_build_agent/__call__.

    Construction, graph assembly and the text entry live here; voice legs
    live on CascadePipeline (built at the end of __init__); two-brain
    routing composes on VoiceAgent.
    ..
    """

    def __init__(self, config: LongRunningConfig | None = None):
        """Build the legs, session store and schedulers. Warms nothing."""
        self.config = config or LongRunningConfig()
        cfg = self.config
        llm_cfg = cfg.llm
        self.delegate_cfg = None
        self.front_llm = None
        self._route_skew: dict = {}
        self.active_model_label: str = "qwen3-0.6B"
        if bool(getattr(cfg, "delegate", False)):
            try:
                dcfg = load_config(getattr(cfg, "delegate_config", None))
                if dcfg.enabled:
                    self.delegate_cfg = dcfg
                    over = worker_llm_overrides(dcfg)
                    if over:
                        llm_cfg = llm_cfg.model_copy(update=over)
            except Exception:
                self.delegate_cfg = None
        # Simple switch: bonsai/qwen run local legs, groq/gemini are
        # stock LangChain chat models used directly (no LlmModel).
        from src.agent.cloud_models import is_cloud_backend, make_chat_model

        _backend = str(getattr(llm_cfg, "backend", "qwen") or "qwen").lower()
        if is_cloud_backend(_backend):
            _named = (
                str(getattr(llm_cfg, "groq_model", "") or "")
                if _backend == "groq"
                else str(getattr(llm_cfg, "gemini_model", "") or "")
            )
            _fallback = str(getattr(llm_cfg, "model", "") or "")
            if not _named and _fallback not in ("", "Qwen/Qwen3-0.6B"):
                _named = _fallback
            self.llm = None
            self.chat_model = make_chat_model(_backend, _named)
        else:
            self.llm = LlmModel(llm_cfg)
            from src.agent.chat_model import LocalChatModel

            self.chat_model = LocalChatModel(llm=self.llm)
        # Voice legs live in the cascade (built here so warm-order stays);
        # the harness keeps the shared tts alias for text-turn audio.
        if self.delegate_cfg is not None:
            try:
                self.front_llm = LlmModel(front_llm_config(self.delegate_cfg))
            except Exception:
                self.front_llm = None
        try:
            self.backend = LocalShellBackend(
                root_dir=cfg.work_dir,
                timeout=int(cfg.exec_timeout_s or cfg.tool_timeout_s),
            )
        except Exception:
            self.backend = None
        # Anchor for MCP relative paths (VOICE_WORKDIR): model-relative
        # paths land where the shell backend runs. Set BEFORE the MCP
        # stdio servers spawn — they snapshot os.environ per process.
        if getattr(cfg, "work_dir", ""):
            os.environ["VOICE_WORKDIR"] = str(cfg.work_dir)
        try:
            self._mcp_client, self._extra_tools = load_extra_tools(cfg.mcp_config)
        except Exception:
            self._mcp_client, self._extra_tools = None, []
        self._mcp_sessions = None
        if hasattr(self.chat_model, "use_mcp_fs"):
            self.chat_model.use_mcp_fs = bool(self._extra_tools)
        self.sessions = JsonSessionMemory(
            sessions_dir=os.path.expanduser(str(cfg.sessions_dir or "~/.voicert")),
            max_turns=cfg.max_session_turns,
            max_age_s=cfg.session_ttl_s,
            max_sessions=cfg.max_sessions,
            max_tokens=getattr(cfg, "memory_tokens", 0) or 0,
        )
        self._episodic = None
        try:
            self._sched = FIFOScheduler(cfg.max_inflight)
        except Exception:
            self._sched = None
        self.missing: list = []
        self._warmed = False
        self._worker_unavailable: str | None = None
        self._worker_warmed = False
        self._worker_eager = bool(getattr(cfg, "worker_eager", False))
        if self.delegate_cfg is not None and bool(
            getattr(self.delegate_cfg, "worker_eager", False)
        ):
            self._worker_eager = True
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, list[str]] = {}
        self._box_lock: asyncio.Lock = asyncio.Lock()
        self._anchored_dir: str | None = None
        self.approvals: dict[str, set[str]] = {}
        self.agent = self._build_agent()

    # ---- long-running harness (agentic loop: build, turns, sessions, box) ----

    def _build_agent(self, confirmer=None, backend=None):
        """Assemble the deep agent: model, shell backend, tools, middleware.

        Layers around each tool call, outermost first: todo tracking,
        observation trimming, permission gate, error hints, result
        verification, optional tool-shortlist injection, and always-on
        in-turn summarization. ``confirmer`` (async (name, args) -> yes/no)
        backs confirm verdicts; None auto-allows (announced nowhere on
        server — the WS gate passes one per turn below). ``backend``
        overrides ``self.backend`` for the turn (sandbox box); None
        keeps it. Execution tools come from the listing layer
        (collect + tag-filter); the semantic top-K stays prompt-side
        and off unless ``use_tool_router`` is set.
        """
        from src.agent.tool_router import build_execution_tools

        turn_backend = backend if backend is not None else self.backend
        middleware = [
            TodoListMiddleware(
                system_prompt=TODO_SYSTEM_PROMPT,
                tool_description=TODO_TOOL_DESCRIPTION,
            ),
            TrimObservationsMiddleware(limit=1500),
            PolicyMiddleware(confirmer=confirmer),
            ErrorHintMiddleware(),
            VerifyMiddleware(),
            *(
                [InjectToolMiddleware(
                    k=int(getattr(self.config, "tool_top_k", 5) or 5))]
                if getattr(self.config, "use_tool_router", False)
                else []
            ),
            SummarizationMiddleware(
                llm=self.llm, backend=turn_backend, model=self.chat_model
            ),
        ]
        middleware = [m for m in middleware if m is not None]

        return create_deep_agent(
            model=self.chat_model,
            backend=turn_backend,
            tools=build_execution_tools(self._extra_tools),
            middleware=middleware,
            system_prompt=SYSTEM_PROMPT
            + " "
            + self._identity_sentence()
            + " "
            + WORKER_STANCE,
        )

    async def open_mcp_sessions(self) -> "VoiceAgent":
        """Open stateful MCP sessions and rebind tools. Returns self."""
        from src.agent.mcp.session import open_sessions

        if getattr(self, "_mcp_sessions", None) is not None:
            return self
        pool = await open_sessions(getattr(self.config, "mcp_config", None))
        self._mcp_sessions = pool
        self._mcp_client = pool
        self._extra_tools = list(pool.tools)
        if hasattr(self.chat_model, "use_mcp_fs"):
            self.chat_model.use_mcp_fs = bool(self._extra_tools)
        try:
            self.agent = self._build_agent()
        except Exception:
            pass
        return self

    async def reopen_mcp_sessions(self) -> "VoiceAgent":
        """Close + reopen sessions (picks up a new VOICE_WORKDIR)."""
        try:
            await self.aclose_mcp()
        except Exception:
            pass
        return await self.open_mcp_sessions()

    async def aclose_mcp(self) -> None:
        """Close the stateful MCP sessions. Never raises."""
        pool, self._mcp_sessions = getattr(self, "_mcp_sessions", None), None
        if pool is not None:
            try:
                await pool.aclose()
            except Exception:
                pass

    async def aclose(self) -> None:
        """Release long-lived handles (MCP sessions). Never raises."""
        try:
            await self.aclose_mcp()
        except Exception:
            pass

    def history(self, session_id: str | None) -> list:
        """Session transcript as role/content dicts, empty when unknown."""
        try:
            return self.sessions.history(session_id) if session_id else []
        except Exception:
            return []

    def export_session(self, session_id: str) -> list:
        """Dump one session transcript for application-level persistence."""
        return self.history(session_id)

    def import_session(self, session_id: str, data: list) -> None:
        """Load role/content dicts into a session. Never raises."""
        try:
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
        """Allowlist a command's first word for the rest of the session."""
        if not session_id or not command:
            return
        self.approvals.setdefault(session_id, set()).add(
            command.strip().split()[0][:40]
        )

    def is_approved(self, session_id: str | None, action) -> bool:
        """Whether the action's command or op is allowlisted this session."""
        if not session_id:
            return False
        allow = self.approvals.get(session_id, set())
        if not allow:
            return False
        cmd = getattr(action, "command", "") or ""
        first = cmd.strip().split()[0] if cmd.strip() else ""
        return first in allow or getattr(action, "op", "") in allow

    async def _admit(self):
        """Take a scheduler slot, timing out when the server is saturated."""
        if self._sched is None:
            return None
        ticket = await asyncio.to_thread(
            self._sched.acquire, True, self.config.queue_timeout_s
        )
        if ticket is None:
            raise TimeoutError(
                f"server saturated ({self._sched.max_concurrency} in flight, "
                f"queue waited {self.config.queue_timeout_s:.0f}s); retry"
            )
        return ticket

    def _release(self, ticket) -> None:
        """Hand a scheduler slot back. Never raises."""
        try:
            if ticket is not None and self._sched is not None:
                self._sched.release(ticket)
        except Exception:
            pass

    def _norm_msg(self, m):
        """Reduce a LangChain message or dict to role, content, calls, kwargs."""
        if isinstance(m, dict):
            return (
                m.get("type") or m.get("role") or "",
                m.get("content", ""),
                m.get("tool_calls"),
                m.get("additional_kwargs") or {},
            )
        return (
            getattr(m, "type", None) or "",
            getattr(m, "content", ""),
            getattr(m, "tool_calls", None),
            getattr(m, "additional_kwargs", None) or {},
        )

    @staticmethod
    def _think_of(ak) -> str:
        """Pull the thinking trace out of message kwargs, else empty."""
        return str(ak.get("thinking", "") or "") if isinstance(ak, dict) else ""

    @staticmethod
    def _resolve_reply(content: str) -> str:
        """Text-only AI content -> reply; legacy done envelopes resolve."""
        txt = str(content or "").strip()
        if not txt:
            return ""
        for cand in (
            parse_terminal_action(txt),
            parse_xml_action(txt),
            parse_bare_tail(txt),
        ):
            if cand is not None and cand.op == "done":
                return cand.reply or ""
        return txt

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
                    self.remember_approval(
                        sid, getattr(action, "command", "") or getattr(action, "op", "")
                    )
                return "always"
            return bool(res) if res is not None else False

        return _wrapper

    def _identity_sentence(self) -> str:
        """Active-model identity line for this turn's system prompt.

        ..
        """
        return identity_sentence(str(getattr(self, "active_model_label", "") or ""))

    def _front_messages(self, text: str, history: list) -> list:
        """Front-model CHAT prompt: one short spoken sentence, no tools.

        Deliberately NOT ``messages_for_terminal``: the front model has no
        terminal, and handing it the op catalogue is how small models start
        emitting ops they cannot run. No tool specs at all, for the same
        reason.
        """
        msgs = [
            {
                "role": "system",
                "content": FRONT_SYSTEM + " " + self._identity_sentence(),
            }
        ]
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
        cap = int(getattr(self.delegate_cfg, "max_task_chars", 1200) or 1200)
        escalate, reason = await decide_escalate(self.front_llm, text)
        if escalate:
            return Route(
                kind="delegate",
                text=str(text),
                reason=reason,
                forced=reason != "boolean",
            )

        reply, reason2 = await self._front_chat(text, history)
        route = parse_delegate(reply, max_task_chars=cap)
        if route is not None:
            return route
        if getattr(self.delegate_cfg, "backstop", False) and is_task_shaped(
            text, int(getattr(self.delegate_cfg, "backstop_min_score", 2) or 2)
        ):
            return Route(
                kind="delegate", text=str(text), reason="backstop", forced=True
            )
        return Route(kind="chat", text=reply, reason=reason2 or reason)

    async def _front_chat(self, text: str, history: list) -> tuple[str, str]:
        """One short spoken reply from the front leg.

        Returns ``(reply, reason)``. Never raises: a dead front leg on the
        chat path is an empty reply, and an empty chat reply falls through
        to the worker in _run_turn rather than answering with silence.
        Delegation-talk ("I'll hand this to a larger model...") is also
        emptied: the front model narrating a handoff instead of answering
        is a miss, and the worker gives the real reply.
        """
        try:
            res = await self.front_llm.generate(
                self._front_messages(text, history), FRONT_CHAT_MAX_TOKENS
            )
            reply = str(getattr(res, "text", "") or "").strip()
            if reply and _DELEGATION_TALK.search(reply):
                return "", "chat-delegating"
            return reply, "chat"
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
            if getattr(self, "chat_model", None) is not None:
                self._worker_warmed = True
                return ""
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
        return (
            str(task),
            WORKER_DOWN_TEMPLATE.format(reason=reason),
            "",
        )

    def _resolve_turn_cwd(self, cwd: str) -> str:
        """Requested cwd -> jailed dir, else config work_dir. Raises _BoxError.

        A rejected dir never mounts and never runs: the turn falls back
        to the configured work dir, and only fails loudly when even the
        fallback is unsafe.
        ..
        """
        sessions_dir = os.path.expanduser(
            str(getattr(self.config, "sessions_dir", "~/.voicert") or "~/.voicert")
        )
        hit = _jail_turn_cwd(cwd, sessions_dir)
        if hit is not None:
            return hit
        base = os.path.abspath(
            os.path.expanduser(str(getattr(self.config, "work_dir", "") or "."))
        )
        if _jail_turn_cwd(base, sessions_dir) is not None:
            return base
        raise _BoxError(f"turn cwd rejected ({str(cwd)[:80]!r}); no safe fallback")

    @asynccontextmanager
    async def _turn_box(self, cwd: str):
        """Jail the turn cwd; optionally run the turn in a fresh container.

        Yields (turn_cwd, turn_backend). With ``config.sandbox`` set, each
        turn gets its own ``--rm`` container rooted at the turn dir, the
        MCP servers reopen into it (``VOICE_WORKDIR`` +
        ``VOICE_SANDBOX_CONTAINER``), and everything is torn down on exit.
        Without it, only a changed workdir re-anchors (fast path needs no
        lock). Setup failures raise _BoxError — the turn errors loudly
        instead of silently running on the host.
        ..
        """
        from src.sandbox.docker import DockerSandboxError, SandboxShellBackend

        turn_cwd = self._resolve_turn_cwd(cwd)
        box_cfg = getattr(self.config, "sandbox", None)
        if (
            box_cfg is None
            and os.environ.get("VOICE_WORKDIR") == turn_cwd
            and not os.environ.get("VOICE_SANDBOX_CONTAINER")
        ):
            yield turn_cwd, self.backend
            return
        async with self._box_lock:
            prev_backend, box, turn_backend = self.backend, None, self.backend
            boxed = False
            try:
                if box_cfg is not None:
                    try:
                        box = box_cfg.create(turn_cwd)
                        box.ensure_running()
                    except DockerSandboxError as exc:
                        raise _BoxError(str(exc)[:300])
                    boxed = True
                    os.environ["VOICE_SANDBOX_CONTAINER"] = box.container
                    turn_backend = SandboxShellBackend(
                        box,
                        timeout=int(
                            getattr(self.config, "exec_timeout_s", 0)
                            or getattr(self.config, "tool_timeout_s", 0)
                            or 30
                        ),
                    )
                os.environ["VOICE_WORKDIR"] = turn_cwd
                self.backend = turn_backend
                await self.reopen_mcp_sessions()
                self._anchored_dir = turn_cwd
                yield turn_cwd, turn_backend
            finally:
                if box is not None:
                    box.close()
                os.environ.pop("VOICE_SANDBOX_CONTAINER", None)
                self.backend = prev_backend
                if boxed:
                    await self.reopen_mcp_sessions()

    async def _agent_invoke(
        self,
        text: str,
        *,
        sid: str | None,
        cwd: str,
        confirm_fn,
        system_suffix: str,
        out_turns: list,
    ):
        """One deep-agent invocation, yielding term/* mid-turn events.

        With delegation on, this is the WORKER leg: it only runs for turns
        the front brain handed over (or forced over via the backstop), and
        it runs the full existing graph.
        """
        raw_confirm = self._confirm_wrapper(confirm_fn, sid)

        async def _policy_confirm(name, args):
            return await raw_confirm({"name": name, "args": dict(args or {})})

        try:
            async with self._turn_box(cwd) as (turn_cwd, turn_backend):
                async for ev in self._invoke_boxed(
                    text,
                    sid=sid,
                    cwd=turn_cwd,
                    backend=turn_backend,
                    confirmer=_policy_confirm,
                    system_suffix=system_suffix,
                    out_turns=out_turns,
                ):
                    yield ev
        except _BoxError as exc:
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
        system_suffix: str,
        out_turns: list,
    ):
        """Boxed worker invocation: jailed cwd + turn backend. Yields events.

        Runs the full existing graph; the caller owns the box lifecycle.
        ..
        """
        try:
            agent = self._build_agent(confirmer=confirmer, backend=backend)
            self.agent = agent
        except Exception as exc:
            yield AgentEvent(
                node="term", kind="error", data={"message": f"agent build: {exc}"[:300]}
            )
            return
        history = self.sessions.history(sid) if sid else []
        mem_ctx = self._memory_context(sid, str(text)) if sid else ""
        lc_msgs = _lc_messages(history) + [
            HumanMessage(content=str(text) + mem_ctx + f"\nCWD: {cwd} SHELL: bash")
        ]
        final_reply = ""
        thinking = ""
        last_think = ""
        live_think_shown = False
        did_work = False
        rl = 10 + int(getattr(self.config, "max_agent_steps", 6) or 6) * 5
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
                        for piece in _chunk_pieces(msg):
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
                        role, content, tool_calls, ak = self._norm_msg(m)
                        if isinstance(content, list):
                            content = " ".join(
                                str(c.get("text", c))
                                for c in content
                                if isinstance(c, dict)
                            )
                        if role in ("ai", "assistant"):
                            think = self._think_of(ak)
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
                                final_reply = self._resolve_reply(content)
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
        thinking2, reply = split_thinking(final_reply)
        if thinking2 and thinking2 != last_think:
            thinking = thinking2
            if not live_think_shown:
                yield AgentEvent(
                    node="term", kind="thinking", data={"text": thinking[:2000]}
                )
            live_think_shown = False
        if not reply and did_work:
            reply = "Done."
        if reply and is_degenerate(reply):
            yield AgentEvent(
                node="term", kind="error", data={"message": "model output degenerate"}
            )
            reply = DEGENERATE_REPLY
        if sid and (text or reply):
            try:
                before = (
                    [m.get("content", "") for m in self.sessions.history(sid)]
                    if getattr(self.config, "memory_store", False)
                    else []
                )
                self.sessions.remember_turn(sid, str(text), reply)
                if before:
                    self._memory_consolidate(sid, before)
            except Exception:
                pass
        out_turns.append((str(text), reply, thinking))

    async def _run_locked(
        self,
        text: str,
        *,
        sid: str | None,
        cwd: str,
        confirm_fn,
        system_suffix: str,
        out_turns: list,
    ):
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
                yield AgentEvent(
                    node="term",
                    kind="queued",
                    data={"position": len(pend), "session_id": sid},
                )
                return
            await lock.acquire()
        try:

            async def _one(t: str):
                """Run one queued text through the turn pipeline."""
                async for ev in self._run_turn(
                    t,
                    sid=sid,
                    cwd=cwd,
                    confirm_fn=confirm_fn,
                    system_suffix=system_suffix,
                    out_turns=out_turns,
                ):
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

    async def _run_turn(
        self,
        text: str,
        *,
        sid: str | None,
        cwd: str,
        confirm_fn,
        system_suffix: str,
        out_turns: list,
    ):
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
                out_turns.append(self._worker_down_notice(str(text), reason))
                return
            async for ev in self._agent_invoke(
                text,
                sid=sid,
                cwd=cwd,
                confirm_fn=confirm_fn,
                system_suffix=system_suffix,
                out_turns=out_turns,
            ):
                yield ev
            return

        history = self.sessions.history(sid) if sid else []
        route = await self._front_decide(text, history)
        self._note_route_disagreement(text, route)

        if getattr(self.delegate_cfg, "emit_route", True):
            yield AgentEvent(
                node="term",
                kind="route",
                data={
                    "brain": "front" if route.kind == "chat" else "worker",
                    "kind": route.kind,
                    "reason": str(route.reason or "")[:80],
                    "forced": bool(route.forced),
                },
            )

        if route.kind == "chat":
            reply = str(route.text or "").strip()
            if not reply:
                reason = await self._ensure_worker()
                if reason:
                    out_turns.append(self._worker_down_notice(str(text), reason))
                    return
                async for ev in self._agent_invoke(
                    text,
                    sid=sid,
                    cwd=cwd,
                    confirm_fn=confirm_fn,
                    system_suffix=system_suffix,
                    out_turns=out_turns,
                ):
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
            task,
            sid=sid,
            cwd=cwd,
            confirm_fn=confirm_fn,
            system_suffix=system_suffix,
            out_turns=out_turns,
        ):
            yield ev
        produced = any(str(reply).strip() for (_t, reply, _th) in out_turns[before:])
        if not produced:
            why = self._worker_missing_reason()
            if why:
                notice = (
                    str(text),
                    WORKER_DOWN_TEMPLATE.format(reason=why),
                    "",
                )
                if len(out_turns) > before:
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
            scored = is_task_shaped(
                text, int(getattr(self.delegate_cfg, "backstop_min_score", 2) or 2)
            )
            said = route.kind == "delegate"
            if scored == said:
                return
            self._route_skew.setdefault(
                "regex_no_model_yes" if said else "model_no_regex_yes", 0
            )
            self._route_skew[
                "regex_no_model_yes" if said else "model_no_regex_yes"
            ] += 1
            log.info(
                "route skew: model=%s regex=%s %.90s", route.kind, scored, str(text)
            )
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

    def _episodic_store(self):
        """Lazy L2 store (None unless memory_store/recall enabled)."""
        if self._episodic is None and (
            getattr(self.config, "memory_store", False)
            or getattr(self.config, "memory_recall", False)
        ):
            try:
                self._episodic = EpisodicStore(
                    os.path.join(
                        getattr(self.config, "memory_dir", "memory"), "episodic.db"
                    )
                )
            except Exception:
                self._episodic = None
        return self._episodic

    def _memory_consolidate(self, sid, before_contents) -> None:
        """Store L1-evicted turns as L2 episodes (extractive summary)."""
        try:
            after = {m.get("content", "") for m in self.sessions.history(sid)}
            dropped = [c for c in (before_contents or []) if c and c not in after]
            if not dropped:
                return
            store = self._episodic_store()
            if store is None:
                return
            summary = summarize_turns([{"role": "user", "content": c} for c in dropped])
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
                        PAST_EPISODES_HEADER
                        + "\n".join("- " + r["summary"][:300] for r in rows)
                    )
            facts = load_facts(
                os.path.join(getattr(self.config, "memory_dir", "memory"), "facts.md")
            )
            if facts:
                parts.append(USER_FACTS_HEADER + format_block(facts))
            return "".join(parts)
        except Exception:
            return ""

    async def run_text(
        self,
        text: str,
        session_id: str | None = None,
        cwd: str | None = None,
        confirm_fn=None,
    ):
        """Run one text turn, yielding term/audio/summary events."""
        if not str(text or "").strip():
            raise ValueError("empty text")
        sid = session_id
        base = cwd or "."
        turns: list = []
        last_reply, last_think = "", ""
        async for ev in self._run_locked(
            str(text),
            sid=sid,
            cwd=base,
            confirm_fn=confirm_fn,
            system_suffix=TEXT_SUFFIX,
            out_turns=turns,
        ):
            yield ev
        for _t, reply, thinking in turns:
            if not reply:
                last_reply, last_think = reply, thinking
                continue
            if is_degenerate(reply):
                yield AgentEvent(
                    node="term",
                    kind="error",
                    data={"message": "model output degenerate"},
                )
                reply = DEGENERATE_REPLY
            last_reply, last_think = reply, thinking
            yield AgentEvent(node="term", kind="chat", data={"reply": reply})
            if (
                getattr(self.config, "speak_text_turns", True)
                and getattr(self, "tts", None) is not None
            ):
                try:
                    out = await self.tts.speak(_speak_head(reply))
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

    async def __call__(self, text: str, session_id: str | None = None) -> str:
        """Text turn on the long-running loop. Returns the final reply.

        Collects run_text events; the last chat event wins. Voice entry
        lives on the cascade; two-brain routing composes on VoiceAgent.
        ..
        """
        reply = ""
        async for ev in self.run_text(str(text), session_id=session_id):
            if ev.kind == "chat":
                reply = str(ev.data.get("reply", "") or reply)
        return reply


