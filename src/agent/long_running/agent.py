"""Long-running agent facade: model legs, stores and public entry points.

Construction, model selection, MCP lifecycle and public session methods
live here. One turn is delegated to ``TurnRunner``; graph assembly to
``builder``; session/memory persistence to middleware; box, worker and
scheduler to their own services.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

from deepagents.backends import LocalShellBackend
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.agent.events import AgentEvent
from src.agent.long_running.box import TurnBox
from src.agent.long_running.builder import build_agent
from src.agent.long_running.recall import AgentMemory
from src.agent.long_running.scheduler import AdmissionQueue, TurnScheduler
from src.agent.long_running.turn import TurnRunner
from src.agent.long_running.worker import WorkerLeg
from src.agent.mcp.client import load_extra_tools
from src.agent.mcp.session import open_sessions
from src.agent.memory import JsonSessionMemory
from src.agent.prompts.voice import identity_sentence
from src.models.chat_model import LocalChatModel
from src.models.llm import LlmConfig, LlmModel
from src.models.provider import ChatModel
from src.paths import VoicertPaths

__all__ = ["LongRunningAgent", "LongRunningConfig"]


def _data_dir() -> str:
    """Local data dir (XDG), shared with the rest of the app."""
    return VoicertPaths.data_dir()


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
    sessions_dir: str = Field(default_factory=lambda: os.path.join(_data_dir(), "sessions"))
    memory_tokens: int = 0
    memory_recall: bool = False
    memory_store: bool = False
    memory_dir: str = Field(default_factory=lambda: os.path.join(_data_dir(), "memory"))
    worker_eager: bool = False
    sandbox: Any = None


class LongRunningAgent:
    """Agentic facade: model legs, sessions, MCP and public entries."""

    def __init__(self, config: LongRunningConfig | None = None):
        """Build the legs, stores and services. Warms nothing."""
        self.config = config or LongRunningConfig()
        cfg = self.config
        llm_cfg = cfg.llm
        self.active_model_label: str = "qwen3-0.6B"
        _backend = str(getattr(llm_cfg, "backend", "qwen") or "qwen").lower()
        if ChatModel.is_cloud_backend(_backend):
            _field = {
                "groq": "groq_model",
                "gemini": "gemini_model",
                "anthropic": "anthropic_model",
                "openai": "openai_model",
                "openrouter": "openrouter_model",
                "deepseek": "deepseek_model",
                "openai_compat": "compat_model",
            }.get(_backend, "")
            _named = str(getattr(llm_cfg, _field, "") or "") if _field else ""
            _fallback = str(getattr(llm_cfg, "model", "") or "")
            if not _named and _fallback not in ("", "Qwen/Qwen3-0.6B"):
                _named = _fallback
            self.llm = None
            self.chat_model = ChatModel.get(
                _backend,
                _named,
                base_url=str(getattr(llm_cfg, "compat_base_url", "") or ""),
            )
        else:
            self.llm = LlmModel(llm_cfg)
            self.chat_model = LocalChatModel(llm=self.llm)
        try:
            self.backend = LocalShellBackend(
                root_dir=cfg.work_dir,
                timeout=int(cfg.exec_timeout_s or cfg.tool_timeout_s),
            )
        except Exception:
            self.backend = None
        if getattr(cfg, "work_dir", ""):
            os.environ["VOICE_WORKDIR"] = str(cfg.work_dir)
        try:
            self._mcp_client, self._extra_tools = load_extra_tools(cfg.mcp_config)
        except Exception:
            self._mcp_client, self._extra_tools = None, []
        self._mcp_sessions = None
        self.sessions = JsonSessionMemory(
            sessions_dir=os.path.expanduser(str(cfg.sessions_dir or _data_dir() + "/sessions")),
            max_turns=cfg.max_session_turns,
            max_age_s=cfg.session_ttl_s,
            max_sessions=cfg.max_sessions,
            max_tokens=getattr(cfg, "memory_tokens", 0) or 0,
        )
        self.memory = AgentMemory(cfg)
        self.worker = WorkerLeg(
            lambda: self.llm, lambda: self.chat_model, eager=cfg.worker_eager
        )
        self.admission = AdmissionQueue(cfg.max_inflight, cfg.queue_timeout_s)
        self.scheduler = TurnScheduler()
        self._box_lock = asyncio.Lock()
        self.box = TurnBox(
            cfg,
            lambda: self.backend,
            self._set_backend,
            self.reopen_mcp_sessions,
            self._box_lock,
        )
        self.runner = TurnRunner(self)
        self.approvals: dict[str, set[str]] = {}
        self._warmed = False
        self.agent = self.build_longrunning_agent()

    @property
    def missing(self) -> list:
        """Leg failures collected during warm or construction."""
        return self.worker.missing

    @missing.setter
    def missing(self, value: list) -> None:
        """Replace the leg-failure list (server model switches reset it)."""
        self.worker.missing = list(value or [])

    def _set_backend(self, backend) -> None:
        """Point the facade at the current turn backend."""
        self.backend = backend

    def build_longrunning_agent(
        self,
        confirmer=None,
        backend=None,
        session_id: str | None = None,
        text: str = "",
        cwd: str = ".",
    ) -> CompiledStateGraph:
        """Assemble the deep agent for a turn or a standby build.

        Session and memory middleware are included whenever the facade
        has a store/service; without a session id they are no-ops.
        ..
        """
        return build_agent(
            config=self.config,
            chat_model=self.chat_model,
            backend=backend if backend is not None else self.backend,
            extra_tools=list(getattr(self, "_extra_tools", []) or []),
            llm=getattr(self, "llm", None),
            identity=self._identity_sentence(),
            sessions=getattr(self, "sessions", None),
            session_id=session_id,
            turn_text=text,
            cwd=cwd,
            memory=getattr(self, "memory", None),
            confirmer=confirmer,
        )

    async def open_mcp_sessions(self) -> "LongRunningAgent":
        """Open stateful MCP sessions and rebind tools. Returns self."""
        if getattr(self, "_mcp_sessions", None) is not None:
            return self
        pool = await open_sessions(getattr(self.config, "mcp_config", None))
        self._mcp_sessions = pool
        self._mcp_client = pool
        self._extra_tools = list(pool.tools)
        try:
            self.agent = self.build_longrunning_agent()
        except Exception:
            pass
        return self

    async def reopen_mcp_sessions(self) -> "LongRunningAgent":
        """Close and reopen sessions (picks up a new VOICE_WORKDIR)."""
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
                if role in ("user", "assistant"):
                    msgs.append((role, content))
            if session_id:
                for role, content in msgs:
                    self.sessions.append(str(session_id), role, content)
        except Exception:
            pass

    def remember_approval(self, session_id: str, command: str) -> None:
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

    def _identity_sentence(self) -> str:
        """Active-model identity line for this turn's system prompt."""
        return identity_sentence(str(getattr(self, "active_model_label", "") or ""))

    async def run_text(
        self,
        text: str,
        session_id: str | None = None,
        cwd: str | None = None,
        confirm_fn=None,
        speak: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        """Run one text turn, yielding term/audio/summary events."""
        async for ev in self.runner.run_text(
            text,
            session_id=session_id,
            cwd=cwd,
            confirm_fn=confirm_fn,
            speak=speak,
        ):
            yield ev

    async def run_events(
        self,
        text: str,
        *,
        sid: str | None,
        cwd: str,
        confirm_fn,
        out_turns: list,
    ) -> AsyncIterator[AgentEvent]:
        """Run one raw turn for the cascade, without speaking or summary."""
        async for ev in self.runner.run(
            text,
            sid=sid,
            cwd=cwd,
            confirm_fn=confirm_fn,
            out_turns=out_turns,
        ):
            yield ev

    async def __call__(self, text: str, session_id: str | None = None) -> str:
        """Text turn on the long-running loop. Returns the final reply."""
        return await self.runner.collect_text(text, session_id=session_id)
