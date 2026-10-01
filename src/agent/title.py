"""Two-word session titles from the front brain.

Ported in spirit from opencode's title generator
(``packages/opencode/src/agent/prompt/title.txt`` + the forked
``ensureTitle`` call in ``session/prompt.ts``): a separate, cheap model call
whose only job is to emit a short name for the conversation.

Two deliberate differences from opencode:

- **The small model is already loaded.** opencode calls
  ``provider.getSmallModel()`` because it has one big model per provider and
  has to go find a cheap one. Here the front leg (``agent.front_llm``,
  Qwen3-0.6B) is a separate instance that is already warm and in VRAM, so
  the title rides it and adds no weights to a 4GB card.
- **The front brain only ever decides, never constructs** — the same
  contract as :data:`~src.agent.delegate.BOOLEAN_SYSTEM`. It is asked for a
  two-word label and nothing else; no tools, no chat history, no task
  string. The failure mode of a 0.6B asked to *write* something is
  narration ("Sure! Here's a title: ..."), which is exactly what
  :func:`parse_title` strips.

The few-shot pairs are not decoration, same as ``BOOLEAN_FEWSHOT``: a small
model handed the format instruction alone reliably adds a preamble and
punctuation, which then has to be parsed away.
"""
from __future__ import annotations

import re

__all__ = ["TITLE_SYSTEM", "TITLE_MAX_TOKENS", "title_messages",
           "parse_title", "generate_title"]

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

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ɏ][0-9A-Za-zÀ-ɏ'+.#-]*")

# Bare cues a model uses when it labels its own answer ("Title: X").
_CUES = {"title", "name", "session", "label", "topic", "answer", "reply"}


def _is_preamble(head: str) -> bool:
    """Is the text before the colon narration rather than part of a title?"""
    words = _WORD_RE.findall(head)
    if not words:
        return False
    if len(words) >= 2:
        return True
    return words[0].lower() in _CUES


def title_messages(text: str) -> list:
    """Prompt for the two-word title.

    Deliberately no tool specs and no chat history, for the same reason as
    :func:`src.agent.delegate.boolean_messages`: the question is about *this*
    utterance, and handing the 0.6B a transcript makes it answer the last
    turn instead of the request in front of it.
    """
    msgs = [{"role": "system", "content": TITLE_SYSTEM}]
    msgs.extend(TITLE_FEWSHOT)
    msgs.append({"role": "user",
                 "content": str(text or "").strip()[:600]})
    return msgs


def parse_title(raw: str, max_words: int = 2, max_chars: int = 40) -> str | None:
    """Two-word title from raw model output, or ``None`` if unusable.

    Strips ``<think>`` traces, preambles ("Sure! Here is a title: ..."),
    quotes and trailing punctuation, then keeps at most ``max_words``
    words. Returns ``None`` rather than a guess so the caller can fall back
    to the offline name instead of printing something meaningless.
    """
    text = re.sub(r"<think>[\s\S]*?</think>", " ", str(raw or ""), flags=re.I)
    text = re.sub(r"<think>[\s\S]*$", " ", text)          # unterminated trace
    text = text.replace("\n", " ").replace("\r", " ")
    # Drop a leading preamble: the model narrating "Sure! Here is a title:
    # Debug errors" names the same topic as "Debug errors". Only when the
    # head reads as narration (two or more words, or a bare cue word) —
    # otherwise a real title like "Fix: wifi driver" loses its own head.
    if ":" in text:
        head, _, tail = text.rpartition(":")
        if _WORD_RE.findall(tail) and _is_preamble(head):
            text = tail
    words = _WORD_RE.findall(text)
    if not words:
        return None
    out = " ".join(words[:max_words]).strip().strip(".,;:!?-—'\"")
    if not out:
        return None
    if len(out) > max_chars:
        out = out[:max_chars].rstrip()
    # A single word is still better than nothing ("Wifi"); an all-punctuation
    # result is not, and was already dropped above.
    return out or None


async def generate_title(front_llm, text: str, *,
                         max_words: int = 2) -> tuple[str | None, str]:
    """Ask the front brain for a two-word title.

    Returns ``(title, reason)``. Never raises: a dead front leg, an
    unparseable answer and an empty utterance all return ``None`` with a
    reason, because the caller falls back to an offline name and a title is
    never worth failing a turn over.
    """
    if front_llm is None:
        return None, "no-front-leg"
    if not str(text or "").strip():
        return None, "empty"
    try:
        res = await front_llm.generate(
            title_messages(text), TITLE_MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001 - a title is never worth an error
        return None, f"error:{type(exc).__name__}"
    title = parse_title(getattr(res, "text", "") or "", max_words=max_words)
    if not title:
        return None, "unparseable"
    return title, "model"