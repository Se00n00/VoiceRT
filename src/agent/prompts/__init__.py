"""Prompts. Single home for every prompt string in the project.

Each submodule owns one family; this package re-exports them all so
callers can do either::

    from src.agent.prompts import TERMINAL_PREAMBLE, BOOLEAN_SYSTEM, ...
    from src.agent.prompts.voice import FRONT_SYSTEM, VOICE_SUFFIX

Layout:

- :mod:`src.agent.prompts.base` — ``SYSTEM_PROMPT`` (voice base)
- :mod:`src.agent.prompts.terminal` — ``TERMINAL_PREAMBLE``,
  ``TERMINAL_TOOL_CATALOGUE``, ``BROWSER_PREAMBLE``
- :mod:`src.agent.prompts.voice` — front/fast/text/voice turn prompts + reply templates
- :mod:`src.agent.prompts.delegate` — boolean routing contract + retired tool spec
- :mod:`src.agent.prompts.title` — two-word session titles
- :mod:`src.agent.prompts.chat` — chat-model repair nudges + repeat guard
- :mod:`src.agent.prompts.memory` — user-facts anchor + headers

Long-running-agent prompts (core system, handoff, todo, memory) live in
``src.agent.long_running.prompts``.

All strings moved here verbatim — behaviour is unchanged. The previous
homes (``src.models.llm``, ``src.agent.voice_agent.agent``, ``src.agent.delegate``,
``src.agent.title``, ``src.agent.long_running.simtask``, ``src.agent.facts``,
``src.models.chat_model``, ``src.agent.long_running.middleware``) re-export their names
from here for backward compatibility.
"""

from src.agent.prompts.base import SYSTEM_PROMPT, WORKER_STANCE
from src.agent.prompts.terminal import (
    BROWSER_PREAMBLE,
    TERMINAL_PREAMBLE,
    TERMINAL_TOOL_CATALOGUE,
)
from src.agent.prompts.voice import (
    DEGENERATE_REPLY,
    FRONT_CHAT_MAX_TOKENS,
    FRONT_SYSTEM,
    TEXT_SUFFIX,
    VOICE_FAST_SYSTEM,
    VOICE_SUFFIX,
    WORKER_DOWN_TEMPLATE,
    identity_sentence,
)
from src.agent.prompts.delegate import (
    BOOLEAN_FEWSHOT,
    BOOLEAN_MAX_TOKENS,
    BOOLEAN_SYSTEM,
    DELEGATE_TOOL,
    DELEGATE_TOOL_NAME,
)
from src.agent.prompts.title import TITLE_FEWSHOT, TITLE_MAX_TOKENS, TITLE_SYSTEM
from src.agent.prompts.chat import (
    ECHO_RETRY,
    GARBLED_RETRY,
    REPEAT_NOTE_TEMPLATE,
    UNPARSEABLE_RETRY,
)
from src.agent.prompts.memory import (
    ANCHOR_LINE,
    FACTS_ANCHOR_LINE,
    FACTS_HEADER,
)

__all__ = [
    "SYSTEM_PROMPT",
    "WORKER_STANCE",
    "TERMINAL_PREAMBLE",
    "TERMINAL_TOOL_CATALOGUE",
    "BROWSER_PREAMBLE",
    "FRONT_SYSTEM",
    "VOICE_FAST_SYSTEM",
    "TEXT_SUFFIX",
    "VOICE_SUFFIX",
    "WORKER_DOWN_TEMPLATE",
    "DEGENERATE_REPLY",
    "FRONT_CHAT_MAX_TOKENS",
    "identity_sentence",
    "BOOLEAN_SYSTEM",
    "BOOLEAN_FEWSHOT",
    "BOOLEAN_MAX_TOKENS",
    "DELEGATE_TOOL",
    "DELEGATE_TOOL_NAME",
    "TITLE_SYSTEM",
    "TITLE_FEWSHOT",
    "TITLE_MAX_TOKENS",
    "GARBLED_RETRY",
    "ECHO_RETRY",
    "UNPARSEABLE_RETRY",
    "REPEAT_NOTE_TEMPLATE",
    "FACTS_ANCHOR_LINE",
    "ANCHOR_LINE",
    "FACTS_HEADER",
]
