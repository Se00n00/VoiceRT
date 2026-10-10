"""Base voice prompt. Single source of truth for SYSTEM_PROMPT.

Moved here from ``src.models.llm`` so every leg (fused, sidecar, paged)
and every caller (``VoiceAgent``, evals, bridges) shares one string.
``src.models.llm`` re-exports this name for backward compatibility.
"""

__all__ = ["SYSTEM_PROMPT"]

SYSTEM_PROMPT = "You are a voice assistant. Reply in one short spoken sentence."
