"""Memory middleware: inject recalled L2 episodes and L3 facts.

The context is resolved once per middleware instance (one turn) and
appended to the model system prompt, never to the user's message. The
store itself stays in ``recall.AgentMemory``.
"""

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware import AgentMiddleware

from src.agent.long_running.recall import AgentMemory

__all__ = ["MemoryMiddleware"]


class MemoryMiddleware(AgentMiddleware):
    """Append this turn's memory context to the system prompt. One job."""

    def __init__(self, memory: AgentMemory, session_id: str | None, text: str):
        """Bind the memory service and query; resolve lazily."""
        self.memory = memory
        self.session_id = session_id
        self.text = str(text or "")
        self._context: str | None = None

    def _context_text(self) -> str:
        """Resolve the recall block once for this turn."""
        if self._context is None:
            try:
                self._context = (
                    self.memory.context(self.session_id, self.text)
                    if self.session_id
                    else ""
                )
            except Exception:
                self._context = ""
        return self._context

    def _request(self, request):
        """Append memory to one model system message. Never raises."""
        context = self._context_text()
        if not context:
            return request
        try:
            system = append_to_system_message(request.system_message, context)
            return request.override(system_message=system)
        except Exception:
            return request

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))
