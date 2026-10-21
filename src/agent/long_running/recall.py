"""L2/L3 memory service for the long-running agent.

Owns the lazy episodic store, recall-context formatting and L1-eviction
consolidation. The graph never sees this class directly: the memory
middleware asks for a context string, and the session middleware calls
``consolidate`` after a turn has been written back.
"""

import os
from typing import TYPE_CHECKING

from src.agent.facts import format_block, load_facts
from src.agent.long_running.episodic import EpisodicStore, summarize_turns
from src.agent.long_running.prompts.memory import (
    PAST_EPISODES_HEADER,
    USER_FACTS_HEADER,
)

if TYPE_CHECKING:
    from src.agent.long_running.agent import LongRunningConfig

__all__ = ["AgentMemory"]


class AgentMemory:
    """L2 episodes plus L3 facts for one agent instance."""

    def __init__(self, config: "LongRunningConfig"):
        """Store the config; the episodic store loads lazily."""
        self.config = config
        self._episodic = None

    def _episodic_store(self):
        """Lazy L2 store (None unless memory_store/recall enabled)."""
        if self._episodic is None and (
            getattr(self.config, "memory_store", False)
            or getattr(self.config, "memory_recall", False)
        ):
            try:
                self._episodic = EpisodicStore(
                    os.path.join(
                        getattr(self.config, "memory_dir", "memory"), "episodic.db"
                    )
                )
            except Exception:
                self._episodic = None
        return self._episodic

    def context(self, sid: str, text: str) -> str:
        """L2 episodes + L3 facts for the current query, or empty.

        ..
        """
        if not getattr(self.config, "memory_recall", False):
            return ""
        try:
            parts = []
            store = self._episodic_store()
            if store is not None:
                rows = store.recall(text, k=3, session=str(sid or ""))
                if rows:
                    parts.append(
                        PAST_EPISODES_HEADER
                        + "\n".join("- " + r["summary"][:300] for r in rows)
                    )
            facts = load_facts(
                os.path.join(getattr(self.config, "memory_dir", "memory"), "facts.md")
            )
            if facts:
                parts.append(USER_FACTS_HEADER + format_block(facts))
            return "".join(parts)
        except Exception:
            return ""

    def consolidate(self, sid: str, before_contents: list, after_contents: list) -> None:
        """Store L1-evicted turns as L2 episodes (extractive summary).

        ..
        """
        try:
            if not getattr(self.config, "memory_store", False):
                return
            store = self._episodic_store()
            if store is None:
                return
            after = {c for c in (after_contents or []) if c}
            dropped = [c for c in (before_contents or []) if c and c not in after]
            if not dropped:
                return
            summary = summarize_turns(
                [{"role": "user", "content": c} for c in dropped]
            )
            if summary:
                store.store(summary, session=str(sid or ""))
        except Exception:
            pass
