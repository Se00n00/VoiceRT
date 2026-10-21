"""Long-running agent middleware: every deepagents hook in one place.

Order (see ``src.agent.long_running.builder``): plan tracking, session
load, memory injection, turn context, observation trimming, permission
gate, error hints, result verification, tool-shortlist injection, and
in-turn summarization.

Layers around each tool call: PolicyMiddleware decides (permission),
the tool node executes, ErrorHintMiddleware converts failures into
guided observations (error handling), VerifyMiddleware checks the
claim against the world (verification).

Each middleware owns one boundary: session read/write, memory context,
turn context, tool gating, error guidance, verification, or summarization.
The builder orders them; callers only pass a flat middleware list.
"""

from langchain.agents.middleware import TodoListMiddleware

from src.agent.long_running.middleware.context import TurnContextMiddleware
from src.agent.long_running.middleware.errors import ErrorHintMiddleware
from src.agent.long_running.middleware.inject import InjectToolMiddleware
from src.agent.long_running.middleware.memory import MemoryMiddleware
from src.agent.long_running.middleware.policy import PolicyMiddleware
from src.agent.long_running.middleware.session import SessionMiddleware
from src.agent.long_running.middleware.summarization import SummarizationMiddleware
from src.agent.long_running.middleware.trim import TrimObservationsMiddleware
from src.agent.long_running.middleware.verify import VerifyMiddleware

__all__ = [
    "TodoListMiddleware",
    "SessionMiddleware",
    "MemoryMiddleware",
    "TurnContextMiddleware",
    "TrimObservationsMiddleware",
    "PolicyMiddleware",
    "ErrorHintMiddleware",
    "VerifyMiddleware",
    "InjectToolMiddleware",
    "SummarizationMiddleware",
]
