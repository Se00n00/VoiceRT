"""User-facts prompt blocks. Single source of truth.

``FACTS_ANCHOR_LINE`` / ``ANCHOR_LINE`` / ``FACTS_HEADER`` moved here
verbatim from ``src.agent.facts`` (which re-exports them). The L2/L3
injection headers used by the long-running agent live in
``src.agent.long_running.prompts.memory``.
"""

__all__ = [
    "FACTS_ANCHOR_LINE",
    "ANCHOR_LINE",
    "FACTS_HEADER",
]

FACTS_ANCHOR_LINE = ("Dated user facts below override older context on conflict; "
               "newer entries win.")
# Legacy name from src.agent.facts — same string, kept for compatibility.
ANCHOR_LINE = FACTS_ANCHOR_LINE
FACTS_HEADER = "# facts — user truths (auto-curated, human-editable)\n"
