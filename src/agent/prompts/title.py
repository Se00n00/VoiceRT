"""Session-title prompts. Single source of truth.

Moved here verbatim from ``src.agent.title`` (which re-exports these
names): the two-word title contract, its few-shot pairs, and its token
budget.
"""

__all__ = ["TITLE_SYSTEM", "TITLE_FEWSHOT", "TITLE_MAX_TOKENS"]

TITLE_SYSTEM = (
    "You name conversations. Output exactly two words naming the topic, and "
    "nothing else. No punctuation, no quotes, no explanation, no greeting. "
    "Use the same language as the user. Never name a tool. If the message is "
    "just a greeting or small talk, name the tone instead.")

# Few-shot, because the format instruction alone is not enough to stop the
# model narrating. Deliberately includes the greeting case: without it the
# model tries to answer "hi" instead of naming it.
TITLE_FEWSHOT = [
    {"role": "user", "content": "debug 500 errors in production"},
    {"role": "assistant", "content": "Production errors"},
    {"role": "user", "content": "refactor the user service"},
    {"role": "assistant", "content": "User service refactor"},
    {"role": "user", "content": "how do I connect postgres to my API"},
    {"role": "assistant", "content": "Postgres API connection"},
    {"role": "user", "content": "add refresh token support to src/auth.ts"},
    {"role": "assistant", "content": "Auth refresh tokens"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Casual greeting"},
    {"role": "user", "content": "why is app.js failing"},
    {"role": "assistant", "content": "App.js failure"},
]

# Two words is the whole output. Anything longer is the model ignoring the
# format rather than thinking, same reasoning as BOOLEAN_MAX_TOKENS.
TITLE_MAX_TOKENS = 12
