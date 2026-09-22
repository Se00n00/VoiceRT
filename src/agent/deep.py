"""DeepAgent for the voice terminal — autonomous, MCP-powered, todos.

Uses :class:`src.agent.chat_model.LocalChatModel` (MiniCPM/Qwen wrapper) as
the model, ``mcp`` terminal tools via :mod:`src.mcp.server`, and
``TodoListMiddleware`` so the agent plans explicitly.

The agent is autonomous: it writes its own todos, calls tools, and
iterates until done. The TUI/bridge streams its events (including
``thinking``) the same way the harness did.
"""
import asyncio
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

from src.agent.chat_model import LocalChatModel
from src.models.llm import LlmModel

__all__ = ["build_deep_agent", "DeepAgentHarness"]


async def build_deep_agent(llm: LlmModel, mcp_tools=None, use_mcp: bool = True):
    """Create the autonomous deep agent, optionally fetching MCP tools.

    Args:
        llm: :class:`LlmModel` (MiniCPM/Qwen wrapper)
        mcp_tools: optional pre-fetched MCP tools (if None and use_mcp True, fetches)
        use_mcp: if True, launch local MCP server and merge its tools
    """
    from src.tools.terminal import TERMINAL_PREAMBLE

    chat_model = LocalChatModel(llm=llm)
    system_prompt = TERMINAL_PREAMBLE + "\n\n" + (
        "You are an autonomous terminal agent. You have a persistent bash shell, "
        "file tools, and a todo list. Always write todos before starting a multi-step task, "
        "update them as you go, and work until done. Prefer read/grep before write/edit. "
        "Use exec for shell, read/write/edit/grep/list for files. Never emit destructive commands; "
        "they are blocked. Be concise and autonomous — do not ask the user for clarification unless blocked."
    )
    tools = list(mcp_tools or [])
    mcp_client = None
    if use_mcp and not tools:
        try:
            import sys as _sys

            from langchain_mcp_adapters.client import MultiServerMCPClient

            mcp_client = MultiServerMCPClient({
                "terminal": {
                    "command": _sys.executable,
                    "args": ["-m", "src.mcp.server"],
                    "transport": "stdio",
                }
            })
            mcp_tools_fetched = await mcp_client.get_tools()
            tools.extend(mcp_tools_fetched)
        except Exception:
            # MCP unavailable — fall back to built-ins
            pass
    agent = create_deep_agent(
        model=chat_model,
        tools=tools,
        middleware=[TodoListMiddleware()],
        system_prompt=system_prompt,
    )
    # Attach client for lifecycle management (caller should keep reference)
    agent._mcp_client = mcp_client  # type: ignore
    return agent


def build_deep_agent_sync(llm: LlmModel, mcp_tools=None):
    """Sync wrapper for tests that don't need MCP."""
    from src.tools.terminal import TERMINAL_PREAMBLE

    chat_model = LocalChatModel(llm=llm)
    system_prompt = TERMINAL_PREAMBLE + "\n\n" + (
        "You are an autonomous terminal agent. You have a persistent bash shell, "
        "file tools, and a todo list. Always write todos before starting a multi-step task, "
        "update them as you go, and work until done. Prefer read/grep before write/edit. "
        "Never emit destructive commands; they are blocked. Be concise."
    )
    tools = mcp_tools or []
    agent = create_deep_agent(
        model=chat_model,
        tools=tools,
        middleware=[TodoListMiddleware()],
        system_prompt=system_prompt,
    )
    return agent


class DeepAgentHarness:
    """Thin harness around the deep agent for bridge/TUI compatibility.

    Mirrors :class:`TerminalHarness.run_turn` but delegates to the deep
    agent graph. Yields ``AgentEvent``-like dicts for the TUI.
    Supports both pre-built agents and lazy MCP-backed creation.
    """

    def __init__(self, llm: LlmModel, mcp_tools=None, agent=None, mcp_client=None):
        self.llm = llm
        self.mcp_tools = mcp_tools or []
        self._mcp_client = mcp_client
        if agent is not None:
            self._agent = agent
        else:
            # Sync fallback without MCP (for tests)
            self._agent = build_deep_agent_sync(llm, self.mcp_tools)

    @classmethod
    async def create(cls, llm: LlmModel, use_mcp: bool = True):
        """Async factory that optionally fetches MCP tools."""
        if use_mcp:
            try:
                agent = await build_deep_agent(llm, mcp_tools=None, use_mcp=True)
                # Retrieve the client from the agent for lifecycle
                mcp_client = getattr(agent, "_mcp_client", None)
                # Extract tools from agent (they are already bound)
                return cls(llm=llm, mcp_tools=[], agent=agent, mcp_client=mcp_client)
            except Exception:
                pass
        # Fallback without MCP
        return cls(llm=llm)

    async def close(self):
        if self._mcp_client:
            try:
                await self._mcp_client.__aexit__(None, None, None)
            except Exception:
                pass

    async def run_turn(self, text: str, session_id: str | None = None, cwd: str | None = None):
        """Run one turn, yielding events. Bridge/TUI can consume directly."""
        from src.agent.events import AgentEvent

        inp = {"messages": [{"role": "user", "content": text}]}
        if cwd:
            inp["messages"][0]["content"] = text + f"\nCWD: {cwd}"
        final_reply = ""
        emitted_chat = False
        did_work = False  # any action/observation this turn
        last_think = ""  # dedupe: never render the same trace twice

        def _think_of(m) -> str:
            ak = m.get("additional_kwargs")
            if isinstance(ak, dict):
                return str(ak.get("thinking", "") or "")
            return ""

        async def _emit_thinking(think: str):
            nonlocal last_think
            think = str(think or "")
            if think and think != last_think:
                last_think = think
                yield AgentEvent(node="term", kind="thinking",
                                 data={"text": think[:2000]})

        async for event in self._agent.astream(inp, stream_mode="updates"):
            if isinstance(event, dict):
                for node, data in event.items():
                    msgs = data.get("messages", []) if isinstance(data, dict) else []
                    for m in msgs if isinstance(msgs, list) else []:
                        # Normalize LangChain message objects to dicts first
                        if not isinstance(m, dict):
                            try:
                                m = {"type": getattr(m, "type", "") or getattr(m, "role", ""),
                                     "content": getattr(m, "content", ""),
                                     "tool_calls": getattr(m, "tool_calls", None),
                                     "additional_kwargs": getattr(m, "additional_kwargs", {})}
                            except Exception:
                                continue
                        role = m.get("type") or m.get("role") or ""
                        content = m.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(str(c.get("text", c)) for c in content if isinstance(c, dict))
                        # Thinking is per-message, not per-tool-call: emit once
                        # before the actions so the TUI shows think→toolcall.
                        if role == "ai":
                            async for ev in _emit_thinking(_think_of(m)):
                                yield ev
                        if role == "ai" and m.get("tool_calls"):
                            for tc in m["tool_calls"]:
                                if isinstance(tc, dict):
                                    tc_name = tc.get("name")
                                    tc_args = tc.get("args") or {}
                                else:  # langchain object form
                                    tc_name = getattr(tc, "name", "?")
                                    tc_args = getattr(tc, "args", {}) or {}
                                yield AgentEvent(node="term", kind="action",
                                                 data={"action": {"action": tc_name, **(tc_args if isinstance(tc_args, dict) else {})}})
                                did_work = True
                        elif role == "ai":
                            if content:
                                final_reply = str(content)
                                emitted_chat = True
                                yield AgentEvent(node="term", kind="chat", data={"reply": final_reply[:2000]})
                        elif role == "tool":
                            did_work = True
                            yield AgentEvent(node="term", kind="observation",
                                             data={"observation": str(content)[:2000]})
                        elif role == "human":
                            # Todo updates are often human-like? ignore
                            pass
        if not emitted_chat and not final_reply and did_work:
            # The turn ran tools but ended with no assistant text
            # (empty final AI message) — without this the TUI shows
            # actions with no visible reply. Surface a closing line.
            final_reply = "Done."
            yield AgentEvent(node="term", kind="chat", data={"reply": final_reply})
        yield AgentEvent(node="term", kind="summary", data={"text": text, "reply": final_reply, "session_id": session_id})
