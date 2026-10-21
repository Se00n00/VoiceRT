"""Graph builder: one place that turns dependencies into a deep agent.

``create_deep_agent`` receives the middleware stack here. Session load,
memory/context injection, tool gates and summarization are separate
classes; the builder only orders them and passes the model, backend,
tools and subagents.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.graph.state import CompiledStateGraph

from src.agent.long_running.middleware import (
    ErrorHintMiddleware,
    InjectToolMiddleware,
    MemoryMiddleware,
    PolicyMiddleware,
    SessionMiddleware,
    SummarizationMiddleware,
    TrimObservationsMiddleware,
    TurnContextMiddleware,
    VerifyMiddleware,
)
from src.agent.long_running.prompts.core import build_system_prompt
from src.agent.long_running.prompts.todo import (
    TODO_SYSTEM_PROMPT,
    TODO_TOOL_DESCRIPTION,
)
from src.agent.long_running.recall import AgentMemory
from src.agent.long_running.specialists import build_specialists
from src.agent.memory import JsonSessionMemory
from src.agent.tool_router import build_execution_tools

if TYPE_CHECKING:
    from src.agent.long_running.agent import LongRunningConfig

__all__ = ["build_agent"]

_AGENTS_PATH = Path(__file__).with_name("AGENTS.md")


def _agents_extension() -> str:
    """User-editable AGENTS.md standing instructions, "" when absent."""
    try:
        return _AGENTS_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def build_agent(
    *,
    config: "LongRunningConfig",
    chat_model: Any,
    backend: Any,
    extra_tools: list,
    llm: Any = None,
    identity: str = "",
    sessions: JsonSessionMemory | None = None,
    session_id: str | None = None,
    turn_text: str = "",
    cwd: str = ".",
    memory: AgentMemory | None = None,
    confirmer: Any = None,
) -> CompiledStateGraph:
    """Assemble the deep agent for one turn or a standby build.

    The order is outermost first: todo, session load, memory, turn
    context, observation trim, policy, error hints, verification,
    optional tool shortlist, then in-turn summarization. Session and
    memory are no-ops without a session id or memory service.
    ..
    """
    middleware = [
        TodoListMiddleware(
            system_prompt=TODO_SYSTEM_PROMPT,
            tool_description=TODO_TOOL_DESCRIPTION,
        ),
        SessionMiddleware(
            sessions, session_id, turn_text,
            on_saved=memory.consolidate if memory is not None else None,
        ) if sessions is not None else None,
        MemoryMiddleware(memory, session_id, turn_text) if memory is not None else None,
        TurnContextMiddleware(cwd),
        TrimObservationsMiddleware(limit=1500),
        PolicyMiddleware(confirmer=confirmer),
        ErrorHintMiddleware(),
        VerifyMiddleware(),
        InjectToolMiddleware(
            k=int(getattr(config, "tool_top_k", 5) or 5)
        ) if getattr(config, "use_tool_router", False) else None,
        SummarizationMiddleware(llm=llm, backend=backend, model=chat_model),
    ]
    middleware = [m for m in middleware if m is not None]

    return create_deep_agent(
        model=chat_model,
        backend=backend,
        tools=build_execution_tools(extra_tools, tags={"Main"}),
        middleware=middleware,
        subagents=build_specialists(
            chat_model,
            extra_tools,
            backend=backend,
            confirmer=confirmer,
        ),
        system_prompt=build_system_prompt(identity, _agents_extension()),
    )
