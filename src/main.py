"""VoiceAgent: autonomous agent with filesystem + shell tools.

One compiled deep-agent over the local model, fed through an LLM queue:

- :meth:`VoiceAgent.__init__` builds the LLM leg, the chat face, the
  execution backend and the agent itself, plus an empty LLM queue.
- :meth:`VoiceAgent._build_agent` compiles the autonomous agent over
  deepagents' built-in tools (ls/read_file/write_file/edit_file/glob/
  grep/execute), running on a local backend rooted at the working dir.
- :meth:`VoiceAgent.__call__` enqueues ``(session_id, text)`` and returns
  the agent's reply. Turns sharing a ``session_id`` see each other's
  in-memory history; ``None`` is a stateless one-shot.

Sessions live in memory only — persisting them (e.g. per-session JSON
dotfiles) is application business, not core: use :meth:`export_session`
/ :meth:`import_session` from the app layer (see ``examples/say_hii.py``).

No warmup — the model backend builds itself on first use. Voice legs
(VAD/STT/TTS) and the rest come later.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.chat_model import LocalChatModel
from src.agent.mcp.client import load_extra_tools
from src.agent.trim import TrimObservationsMiddleware
from src.models.llm import LlmConfig, LlmModel, SYSTEM_PROMPT, split_thinking

__all__ = ["VoiceAgentConfig", "VoiceAgent"]


@dataclass(frozen=True)
class VoiceAgentConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    max_queue: int = 16
    recursion_limit: int = 10
    max_session_turns: int = 8
    max_sessions: int = 1000
    work_dir: str = "."
    exec_timeout_s: float = 30.0
    mcp_config: str | None = None


class VoiceAgent:
    def __init__(self, config: VoiceAgentConfig | None = None):
        self.config = config or VoiceAgentConfig()
        cfg = self.config
        self.llm = LlmModel(cfg.llm)
        self.chat_model = LocalChatModel(llm=self.llm)
        self.backend = LocalShellBackend(
            root_dir=cfg.work_dir, timeout=int(cfg.exec_timeout_s))
        self._mcp_client, self._extra_tools = load_extra_tools(cfg.mcp_config)
        self._sessions: dict[str, list] = {}
        self._LLMQueue: asyncio.Queue = asyncio.Queue(
            maxsize=max(int(self.config.max_queue), 1))
        self._worker: asyncio.Task | None = None
        self.agent = self._build_agent(self._extra_tools)

    def _build_agent(self, extra_tools):
        return create_deep_agent(
            model=self.chat_model,
            backend=self.backend,
            tools=list(extra_tools),
            middleware=[
                TodoListMiddleware(
                    system_prompt="Plan multi-step work with write_todos; "
                                  "skip it for single replies.",
                    tool_description="Track multi-step work "
                                     "(one call per turn).",
                ),
                TrimObservationsMiddleware(limit=1500),
            ],
            system_prompt=SYSTEM_PROMPT,
        )

    async def _run(self, text: str, session_id: str | None) -> str:
        hist = self._sessions.get(session_id, []) if session_id else []
        res = await self.agent.ainvoke(
            {"messages": [*hist, HumanMessage(content=str(text))]},
            config={"recursion_limit": self.config.recursion_limit},
        )
        msgs = res.get("messages", []) if isinstance(res, dict) else []
        reply, thinking = "", ""
        for m in reversed(list(msgs or [])):
            if isinstance(m, dict):
                role = m.get("type") or m.get("role") or ""
                content = m.get("content", "")
                ak = m.get("additional_kwargs") or {}
            else:
                role = getattr(m, "type", None) or ""
                content = getattr(m, "content", "")
                ak = getattr(m, "additional_kwargs", None) or {}
            if role not in ("ai", "assistant"):
                continue
            if isinstance(content, list):
                content = " ".join(
                    str(c.get("text", c)) for c in content
                    if isinstance(c, dict))
            thinking = str(ak.get("thinking", "") or "") \
                if isinstance(ak, dict) else ""
            reply = str(content or "")
            break
        try:
            thinking2, answer = split_thinking(reply)
        except Exception:
            thinking2, answer = thinking, reply
        reply = str(answer or reply or "").strip()
        if session_id:
            window = [*hist,
                      HumanMessage(content=str(text)),
                      AIMessage(content=reply)]
            keep = max(int(self.config.max_session_turns), 1) * 2
            self._sessions[session_id] = window[-keep:]
            if len(self._sessions) > max(int(self.config.max_sessions), 1):
                self._sessions.pop(next(iter(self._sessions)))
        return reply

    def history(self, session_id: str | None) -> list:
        out = []
        for m in self._sessions.get(session_id, []) if session_id else []:
            if isinstance(m, dict):
                role = m.get("type") or m.get("role") or ""
                content = m.get("content", "")
            else:
                kind = getattr(m, "type", None) or ""
                role = "assistant" if kind == "ai" else (
                    "user" if kind == "human" else kind)
                content = getattr(m, "content", "")
            out.append({"role": role, "content": str(content or "")})
        return out

    def export_session(self, session_id: str) -> list:
        return self.history(session_id)

    def import_session(self, session_id: str, data: list) -> None:
        try:
            msgs: list[Any] = []
            for m in data if isinstance(data, list) else []:
                if not isinstance(m, dict):
                    continue
                role, content = m.get("role"), str(m.get("content", "") or "")
                if role == "assistant":
                    msgs.append(AIMessage(content=content))
                elif role == "user":
                    msgs.append(HumanMessage(content=content))
            keep = max(int(self.config.max_session_turns), 1) * 2
            self._sessions[str(session_id)] = msgs[-keep:]
        except Exception:
            pass

    async def _serve(self) -> None:
        while True:
            sid, text, fut = await self._LLMQueue.get()
            try:
                reply = await self._run(text, sid)
                if not fut.cancelled():
                    fut.set_result(reply)
            except Exception as exc:
                if not fut.cancelled():
                    fut.set_exception(exc)
            finally:
                self._LLMQueue.task_done()

    async def __call__(self, text: str,
                       session_id: str | None = None) -> str:
        loop = asyncio.get_running_loop()
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._serve())
        fut = loop.create_future()
        await self._LLMQueue.put((session_id, str(text), fut))
        return await fut
