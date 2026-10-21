"""Long-running agent prompts. Everything the harness prompt-side owns.

- :mod:`src.agent.long_running.prompts.core` — fixed core system prompt +
  the user-editable ``AGENTS.md`` merge.
- :mod:`src.agent.long_running.prompts.handoff` — specialist role prompts.
- :mod:`src.agent.long_running.prompts.todo` — TodoListMiddleware contract.
- :mod:`src.agent.long_running.prompts.memory` — L2/L3 memory injection headers.
- :mod:`src.agent.long_running.prompts.sim` — legacy Sim-Agent templates.

Shared prompts (base, voice, terminal, chat, memory) stay in
``src.agent.prompts``.
"""

from src.agent.long_running.prompts.core import (
    AGENTS_HEADER,
    CORE_SYSTEM_PROMPT,
    build_system_prompt,
)
from src.agent.long_running.prompts.handoff import (
    SPECIALIST_DESCRIPTIONS,
    SPECIALIST_SLIVER,
    SPECIALIST_WARNINGS,
)
from src.agent.long_running.prompts.memory import (
    PAST_EPISODES_HEADER,
    USER_FACTS_HEADER,
)
from src.agent.long_running.prompts.sim import (
    FINAL_COMPOSE_TEMPLATE,
    MAX_SUBTASKS,
    PLAN_PROMPT,
    SUBTASK_ENVELOPE_TEMPLATE,
    SUBTASK_HEAD_TEMPLATE,
    SYSTEM_SLIVER,
)
from src.agent.long_running.prompts.todo import TODO_SYSTEM_PROMPT, TODO_TOOL_DESCRIPTION

__all__ = [
    "CORE_SYSTEM_PROMPT",
    "AGENTS_HEADER",
    "build_system_prompt",
    "SPECIALIST_SLIVER",
    "SPECIALIST_WARNINGS",
    "SPECIALIST_DESCRIPTIONS",
    "TODO_SYSTEM_PROMPT",
    "TODO_TOOL_DESCRIPTION",
    "PAST_EPISODES_HEADER",
    "USER_FACTS_HEADER",
    "SYSTEM_SLIVER",
    "PLAN_PROMPT",
    "MAX_SUBTASKS",
    "SUBTASK_HEAD_TEMPLATE",
    "FINAL_COMPOSE_TEMPLATE",
    "SUBTASK_ENVELOPE_TEMPLATE",
]
