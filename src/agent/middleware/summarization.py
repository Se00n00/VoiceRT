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


def SummarizationMiddleware(*, llm=None, backend=None, model=None):
    """Build the summarization middleware, or None when unavailable."""
    try:
        if model is None and llm is not None and hasattr(llm, "config"):
            from src.agent.chat_model import LocalChatModel

            backend_id = str(getattr(llm.config, "backend", ""))
            ctx = int(CONTEXT_TOKENS.get(backend_id, 8192))
            model = LocalChatModel(llm=llm, no_tools=True)
        elif model is not None:
            ctx = int(CONTEXT_TOKENS.get("groq", 8192))
        else:
            return None
        return _DeepSummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=("tokens", max(2048, int(ctx * 0.9))),
            keep=("messages", 6),
            token_counter=count_messages,
            trim_tokens_to_summarize=1500,
        )
    except Exception:
        return None
