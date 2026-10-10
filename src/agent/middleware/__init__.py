"""Agent middleware: every deepagents hook in one place.

Order (see ``src.main.VoiceAgent._build_agent``): plan tracking,
observation trimming, policy gate, result verification, tool-shortlist
injection, in-turn summarization.

The semantic router instance also lives here: one process-wide lazy
singleton shared by every agent (the router is stateless apart from
its lazily-loaded embedding model, which is itself lock-guarded).
Callers only ever receive the middleware list.
"""

from langchain.agents.middleware import TodoListMiddleware

from src.agent.middleware.inject import InjectToolMiddleware
from src.agent.middleware.policy import PolicyMiddleware
from src.agent.middleware.summarization import SummarizationMiddleware
from src.agent.middleware.trim import TrimObservationsMiddleware
from src.agent.middleware.verify import VerifyMiddleware

__all__ = [
    "TodoListMiddleware",
    "TrimObservationsMiddleware",
    "PolicyMiddleware",
    "VerifyMiddleware",
    "InjectToolMiddleware",
    "SummarizationMiddleware",
]
