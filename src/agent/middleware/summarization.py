"""In-turn compaction guard (returns None when unavailable).

deepagents' SummarizationMiddleware with OUR estimator and a
tool-stripped summarizer sharing the warmed leg (no extra warm,
thinking capped by the leg config). Scope is strictly in-turn
growth: trigger near ctx, keep recent steps. Cross-turn memory
stays in sessions/L2/L3 — this middleware is amnesiac across
turns (the graph is rebuilt per turn), so it must never be the
only compressor. File offload stays OFF for the same reason
(two writers, one summary).
"""
from deepagents.middleware.summarization import (
    SummarizationMiddleware as _DeepSummarizationMiddleware,
)

from src.agent.budget import CONTEXT_TOKENS, count_messages
from src.agent.chat_model import LocalChatModel

__all__ = ["SummarizationMiddleware"]


def SummarizationMiddleware(*, llm, backend):
    """Build the summarization middleware, or None when unavailable."""
    try:
        backend_id = str(getattr(
            getattr(llm, "config", None), "backend", ""))
        ctx = int(CONTEXT_TOKENS.get(backend_id, 8192))
        summarizer = LocalChatModel(llm=llm, no_tools=True)
        return _DeepSummarizationMiddleware(
            model=summarizer,
            backend=backend,
            trigger=("tokens", max(2048, int(ctx * 0.9))),
            keep=("messages", 6),
            token_counter=count_messages,
            trim_tokens_to_summarize=1500,
        )
    except Exception:
        return None
