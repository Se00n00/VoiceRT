"""Sentence-boundary splitter and SSE event formatter for streaming turns."""
import json
import re

# Same guard set as the VOICE prototype: don't split after common abbreviations.
SPLIT = re.compile(r"(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!St)(?<=[.!?])\s+")


def split_sentences(text):
    """Split text into sentences; returns [] for blank input."""
    if not text or not text.strip():
        return []
    return [p for p in SPLIT.split(text.strip()) if p.strip()]


def is_complete_sentence(text):
    """True when text ends at a sentence boundary with nothing pending."""
    if not text or not text.strip():
        return False
    parts = SPLIT.split(text)
    return len(parts) > 1 and not parts[-1].strip()


class SentenceSplitter:
    """Incremental splitter: push() yields newly completed sentences."""

    def __init__(self):
        self._buf = ""

    def push(self, piece):
        """Append streamed text; return sentences completed by this piece."""
        self._buf += piece or ""
        parts = SPLIT.split(self._buf)
        self._buf = parts[-1]
        return [s.strip() for s in parts[:-1] if s.strip()]

    def flush(self):
        """Return and clear any trailing partial sentence ('' if none)."""
        tail, self._buf = self._buf.strip(), ""
        return tail

    @property
    def buffered(self):
        """Current unflushed remainder."""
        return self._buf

    def reset(self):
        """Clear buffered text."""
        self._buf = ""


def format_sse(event, data):
    """Format one Server-Sent Events frame.

    `data` may be a str (sent verbatim) or any JSON-serializable object.
    Multi-line payloads are split into multiple `data:` lines per spec.
    """
    if not isinstance(data, str):
        data = json.dumps(data)
    lines = [f"event: {event}"] if event else []
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def done_frame():
    """Standard [DONE] terminator frame for SSE chat streams."""
    return "data: [DONE]\n\n"
