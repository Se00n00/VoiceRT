"""Turn-context middleware: put cwd and shell in the system prompt.

The model needs the working directory and shell for tool calls, but that
context is not part of the user's utterance. Appending it to the system
message keeps the transcript clean and survives every model call.
"""

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware import AgentMiddleware

__all__ = ["TurnContextMiddleware"]


class TurnContextMiddleware(AgentMiddleware):
    """Append the jailed turn directory and shell to the system prompt."""

    def __init__(self, cwd: str, shell: str = "bash"):
        """Build the static context line for this turn."""
        self.text = f"Working directory: {str(cwd or '.')}\nShell: {shell}"

    def _request(self, request):
        """Append turn context to one model system message."""
        try:
            system = append_to_system_message(request.system_message, self.text)
            return request.override(system_message=system)
        except Exception:
            return request

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))
