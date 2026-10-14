"""Agent middleware: every deepagents hook in one place.

Order (see ``LongRunningAgent._build_agent``): plan tracking,
observation trimming, permission gate, error hints, result
verification, tool-shortlist injection, in-turn summarization.

Layers around each tool call: PolicyMiddleware decides (permission),
the tool node executes, ErrorHintMiddleware converts failures into
guided observations (error handling), VerifyMiddleware checks the
claim against the world (verification).

The semantic router instance also lives here: one process-wide lazy
singleton shared by every agent (the router is stateless apart from
its lazily-loaded embedding model, which is itself lock-guarded).
Callers only ever receive the middleware list.
"""

from langchain.agents.middleware import TodoListMiddleware

from src.agent.middleware.errors import ErrorHintMiddleware
from src.agent.middleware.inject import InjectToolMiddleware
from src.agent.middleware.policy import PolicyMiddleware
from src.agent.middleware.summarization import SummarizationMiddleware
from src.agent.middleware.trim import TrimObservationsMiddleware
from src.agent.middleware.verify import VerifyMiddleware

__all__ = [
    "TodoListMiddleware",
    "TrimObservationsMiddleware",
    "PolicyMiddleware",
    "ErrorHintMiddleware",
    "VerifyMiddleware",
    "InjectToolMiddleware",
    "SummarizationMiddleware",
]
