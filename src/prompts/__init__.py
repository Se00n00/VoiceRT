"""Prompts. Single home for every prompt string in the project.

Each submodule owns one family; this package re-exports them all so
callers can do either::

    from src.prompts import TERMINAL_PREAMBLE, BOOLEAN_SYSTEM, ...
    from src.prompts.voice import FRONT_SYSTEM, VOICE_SUFFIX

Layout:

- :mod:`src.prompts.base` — ``SYSTEM_PROMPT`` (voice base)
- :mod:`src.prompts.terminal` — ``TERMINAL_PREAMBLE``, ``BROWSER_PREAMBLE``
- :mod:`src.prompts.voice` — front/fast/text/voice turn prompts + reply templates
- :mod:`src.prompts.delegate` — boolean routing contract + retired tool spec
- :mod:`src.prompts.title` — two-word session titles
- :mod:`src.prompts.sim` — Sim-Agent sliver / plan / subtask templates
- :mod:`src.prompts.middleware` — todo-list middleware contract
- :mod:`src.prompts.chat` — chat-model repair nudges + repeat guard
- :mod:`src.prompts.memory` — L2/L3 memory injection blocks

All strings moved here verbatim — behaviour is unchanged. The previous
homes (``src.models.llm``, ``src.main``, ``src.agent.delegate``,
``src.agent.title``, ``src.agent.simtask``, ``src.agent.facts``,
``src.agent.chat_model``, ``src.agent.middleware``) re-export their names
from here for backward compatibility.
"""

from src.prompts.base import SYSTEM_PROMPT
from src.prompts.terminal import BROWSER_PREAMBLE, TERMINAL_PREAMBLE
from src.prompts.voice import (
    DEGENERATE_REPLY,
    FRONT_CHAT_MAX_TOKENS,
    FRONT_SYSTEM,
    TEXT_SUFFIX,
    VOICE_FAST_SYSTEM,
    VOICE_SUFFIX,
    WORKER_DOWN_TEMPLATE,
)
from src.prompts.delegate import (
    BOOLEAN_FEWSHOT,
    BOOLEAN_MAX_TOKENS,
    BOOLEAN_SYSTEM,
    DELEGATE_TOOL,
    DELEGATE_TOOL_NAME,
)
from src.prompts.title import TITLE_FEWSHOT, TITLE_MAX_TOKENS, TITLE_SYSTEM
from src.prompts.sim import (
    FINAL_COMPOSE_TEMPLATE,
    MAX_SUBTASKS,
    PLAN_PROMPT,
    SUBTASK_ENVELOPE_TEMPLATE,
    SUBTASK_HEAD_TEMPLATE,
    SYSTEM_SLIVER,
)
from src.prompts.middleware import TODO_SYSTEM_PROMPT, TODO_TOOL_DESCRIPTION
from src.prompts.chat import (
    ECHO_RETRY,
    GARBLED_RETRY,
    REPEAT_NOTE_TEMPLATE,
    UNPARSEABLE_RETRY,
)
from src.prompts.memory import (
    ANCHOR_LINE,
    FACTS_ANCHOR_LINE,
    FACTS_HEADER,
    PAST_EPISODES_HEADER,
    USER_FACTS_HEADER,
)

__all__ = [
    "SYSTEM_PROMPT",
    "TERMINAL_PREAMBLE",
    "BROWSER_PREAMBLE",
    "FRONT_SYSTEM",
    "VOICE_FAST_SYSTEM",
    "TEXT_SUFFIX",
    "VOICE_SUFFIX",
    "WORKER_DOWN_TEMPLATE",
    "DEGENERATE_REPLY",
    "FRONT_CHAT_MAX_TOKENS",
    "BOOLEAN_SYSTEM",
    "BOOLEAN_FEWSHOT",
    "BOOLEAN_MAX_TOKENS",
    "DELEGATE_TOOL",
    "DELEGATE_TOOL_NAME",
    "TITLE_SYSTEM",
    "TITLE_FEWSHOT",
    "TITLE_MAX_TOKENS",
    "SYSTEM_SLIVER",
    "PLAN_PROMPT",
    "MAX_SUBTASKS",
    "SUBTASK_HEAD_TEMPLATE",
    "FINAL_COMPOSE_TEMPLATE",
    "SUBTASK_ENVELOPE_TEMPLATE",
    "TODO_SYSTEM_PROMPT",
    "TODO_TOOL_DESCRIPTION",
    "GARBLED_RETRY",
    "ECHO_RETRY",
    "UNPARSEABLE_RETRY",
    "REPEAT_NOTE_TEMPLATE",
    "FACTS_ANCHOR_LINE",
    "ANCHOR_LINE",
    "FACTS_HEADER",
    "PAST_EPISODES_HEADER",
    "USER_FACTS_HEADER",
]
