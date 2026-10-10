"""Chat-model repair nudges. Single source of truth.

Moved here verbatim from ``src.agent.chat_model.LocalChatModel``: the
one-retry nudges for garbled / echoed / unparsable tool calls, and the
repeat-guard note that steers the model off a third identical read-only
call. ``LocalChatModel`` imports these; behaviour is unchanged.
"""

__all__ = [
    "GARBLED_RETRY",
    "ECHO_RETRY",
    "UNPARSEABLE_RETRY",
    "REPEAT_NOTE_TEMPLATE",
    "VERIFY_RETRY",
]

GARBLED_RETRY = (
    "That reply was garbled. Answer again: ONLY one JSON "
    "action or one short sentence."
)

ECHO_RETRY = (
    "Do not repeat your thinking as the reply. Reply with "
    "EITHER one short chat sentence OR exactly one function "
    "call, nothing else."
)

UNPARSEABLE_RETRY = (
    "Your tool call didn't parse (invalid JSON?). Reply "
    "with EXACTLY one valid function call, nothing else."
)

# Repeat guard (read-only op called twice with identical args and evidence
# in hand). Formatted with ``name``: the op that would repeat a third time.
REPEAT_NOTE_TEMPLATE = (
    "I already called {name} with these exact arguments and "
    "have the results above. Do NOT call it again — answer "
    "the user now from those results, in one short chat "
    "sentence, no tool call."
)

VERIFY_RETRY = (
    "Three tool results in a row did not match what the tools "
    "claimed. Stop and restate: what observable evidence do you "
    "have, and which single check (list_directory/read_file) "
    "settles it next?"
)
