"""Spoken-reply shaping and process VRAM readout for the agent.

Pure text caps used by both text and voice entries; ``vram_mb`` is the
one guarded runtime read. No model or graph state is touched.
"""

from src.models.runtime import max_allocated_mb

__all__ = ["MAX_SPOKEN_WORDS", "cap_words", "speak_head", "vram_mb"]

MAX_SPOKEN_WORDS = 500


def cap_words(text: str, limit: int = MAX_SPOKEN_WORDS) -> str:
    """Hard word cap for spoken replies (model instruction is advisory).

    ..
    """
    words = str(text or "").split()
    if len(words) <= limit:
        return str(text or "")
    return " ".join(words[:limit])


def speak_head(full: str, limit: int = 280) -> str:
    """Short speakable head of a reply (full text stays on screen).

    ..
    """
    t = " ".join(str(full or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    dot = cut.rfind(". ")
    head = cut[: dot + 1] if dot > 120 else cut
    return head.strip() + " … full output is on screen."


def vram_mb() -> float | None:
    """Current process GPU allocation in MB, or None without CUDA.

    ..
    """
    try:
        return float(max_allocated_mb())
    except Exception:
        return None
