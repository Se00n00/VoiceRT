"""Agent: event currency, session memory, terminal tools.

The turn loop lives in :class:`src.main.VoiceAgent` (deepagents over the
local model). This package holds the pieces: per-node stream events,
session stores (RAM + JSON-file), and the terminal toolset.
"""
from src.agent.events import AgentEvent, NODES
from src.agent.memory import (
    JsonSessionMemory,
    LangChainSessionMemory,
    LCSessionHistory,
)

__all__ = [
    "AgentEvent",
    "NODES",
    "JsonSessionMemory",
    "LangChainSessionMemory",
    "LCSessionHistory",
]
