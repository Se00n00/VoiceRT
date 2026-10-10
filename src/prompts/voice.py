"""Voice/text turn prompts. Single source of truth.

Moved here from ``src.main`` (``_FRONT_SYSTEM``, ``_VOICE_FAST_SYSTEM``,
``_TEXT_SUFFIX``, ``_VOICE_SUFFIX``) and ``src.models.llm``
(``SYSTEM_PROMPT``, via :mod:`src.prompts.base`). ``src.main`` keeps
backward-compatible aliases (``_FRONT_SYSTEM`` etc.) re-exported from
here so existing imports keep working.

Also owns the small user-facing reply templates used by ``VoiceAgent``:
the worker-down notice and the degenerate-output fallback.
"""

from src.prompts.base import SYSTEM_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "FRONT_SYSTEM",
    "VOICE_FAST_SYSTEM",
    "TEXT_SUFFIX",
    "VOICE_SUFFIX",
    "WORKER_DOWN_TEMPLATE",
    "DEGENERATE_REPLY",
    "FRONT_CHAT_MAX_TOKENS",
]

FRONT_SYSTEM = (
    "You are a voice assistant. The user is talking, not asking you to do "
    "anything on their computer. Reply in plain speech, no tools, no "
    "preamble, no lists. Do not speak more than 500 words. "
    "You ARE the assistant answering right now: never mention delegation, "
    "larger or bigger models, handoffs, or escalation."
)

VOICE_FAST_SYSTEM = (
    "You are a voice assistant. Reply in plain speech, "
    "no tools, no preamble, just the reply. Do not speak "
    "more than 500 words."
)

TEXT_SUFFIX = (
    "\n\nWork until done: use tools step by step, then give one short final reply."
)

VOICE_SUFFIX = "\n\n" + SYSTEM_PROMPT

# "I handed that to the worker agent, but it could not start, so nothing
# was done: {reason}" — VoiceAgent._worker_down_notice.
WORKER_DOWN_TEMPLATE = (
    "I handed that to the worker agent, but it could not "
    "start, so nothing was done: {reason}"
)

# Fallback chat reply when the model output is degenerate.
DEGENERATE_REPLY = "Sorry — I garbled that. Try rephrasing."

# One short spoken reply from the front leg (VoiceAgent._front_chat).
FRONT_CHAT_MAX_TOKENS = 96
