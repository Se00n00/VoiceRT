"""Three-tier memory prompt blocks. Single source of truth.

``FACTS_ANCHOR_LINE`` / ``ANCHOR_LINE`` / ``FACTS_HEADER`` moved here
verbatim from ``src.agent.facts`` (which re-exports them).
``PAST_EPISODES_HEADER`` / ``USER_FACTS_HEADER`` are the L2/L3 injection
headers built inline in ``VoiceAgent._memory_context`` — now named
constants so the injection point reads instead of guessing.
"""

__all__ = [
    "FACTS_ANCHOR_LINE",
    "ANCHOR_LINE",
    "FACTS_HEADER",
    "PAST_EPISODES_HEADER",
    "USER_FACTS_HEADER",
]

FACTS_ANCHOR_LINE = ("Dated user facts below override older context on conflict; "
               "newer entries win.")
# Legacy name from src.agent.facts — same string, kept for compatibility.
ANCHOR_LINE = FACTS_ANCHOR_LINE
FACTS_HEADER = "# facts — user truths (auto-curated, human-editable)\n"

PAST_EPISODES_HEADER = (
    "\n[Past episodes — USE these to answer when "
    "relevant; prefer them over guessing]\n"
)

USER_FACTS_HEADER = "\n[User facts]\n"
