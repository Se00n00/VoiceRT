"""Deep-agent middleware prompts. Single source of truth.

Custom TodoListMiddleware contract for this harness (small local
models, read-first file work, voice turns): one prompt for the whole
todo discipline (system side) plus one for the write_todos tool
itself (tool side). Both builders (long-running agent, console) use
these — never the library defaults.
"""

__all__ = ["TODO_SYSTEM_PROMPT", "TODO_TOOL_DESCRIPTION"]

TODO_SYSTEM_PROMPT = (
    "## Plan discipline (whole task)"
    "Plan multi-step work with write_todos BEFORE acting: one item per "
    "concrete step (look, change, check), each a few words. Exactly ONE "
    "item in_progress at a time; mark it completed the moment its work "
    "is done — never batch completions, never rewrite history. "
    "Single-step replies and plain chat need no plan: answer directly. "
    "If the work changes shape mid-turn, revise the list first, then "
    "continue. The plan tracks the work; it never delivers the answer — "
    "the final message after the last write_todos call carries the "
    "result the user asked for."
)

TODO_TOOL_DESCRIPTION = (
    "Track the current plan (one call per turn). Each item: content "
    "(what, concretely) plus status pending/in_progress/completed. "
    "Rules per call: put the step you are about to do in_progress; "
    "flip it to completed only when its evidence is in hand; append "
    "follow-ups the moment you discover them; drop dead items. "
    "Use it when the task needs 3+ tool steps; skip it for single "
    "lookups and chat."
)
