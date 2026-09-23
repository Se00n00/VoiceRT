"""Cap tool-result size before the model sees it. Never raises.

Tool outputs (file reads, greps, shell output) are unbounded until
deepagents' far-away eviction thresholds. On a small local model every
extra char costs context, so this middleware truncates long
``ToolMessage`` text to a head + omission note. Short results pass
through untouched.
"""
from langchain.agents.middleware import AgentMiddleware

__all__ = ["TrimObservationsMiddleware"]


class TrimObservationsMiddleware(AgentMiddleware):
    """Truncate long tool results. ``limit`` is max kept chars."""

    def __init__(self, limit: int = 1500):
        self.limit = max(int(limit), 1)

    def _trimmed(self, messages: list) -> list:
        from langchain_core.messages import ToolMessage

        out = []
        for m in messages:
            try:
                if (isinstance(m, ToolMessage)
                        and isinstance(m.content, str)
                        and len(m.content) > self.limit):
                    keep = m.content[:self.limit]
                    note = (f"\n…[{len(m.content) - self.limit} chars "
                            f"omitted]")
                    m = m.model_copy(update={"content": keep + note})
            except Exception:
                pass
            out.append(m)
        return out

    def _apply(self, request):
        """Trimmed request copy (never raises, never mutates in place)."""
        try:
            messages = self._trimmed(list(request.messages or []))
        except Exception:
            return request
        try:
            return request.override(messages=messages)
        except Exception:
            return request

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))
