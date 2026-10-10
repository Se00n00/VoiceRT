"""Deep-agent middleware prompts. Single source of truth.

Moved here verbatim from ``src.agent.middleware`` (``build_middleware``):
the todo-list planner contract shown to the model.
"""

__all__ = ["TODO_SYSTEM_PROMPT", "TODO_TOOL_DESCRIPTION"]

TODO_SYSTEM_PROMPT = (
    "Plan multi-step work with write_todos; "
    "skip it for single replies."
)

TODO_TOOL_DESCRIPTION = (
    "Track multi-step work "
    "(one call per turn)."
)
