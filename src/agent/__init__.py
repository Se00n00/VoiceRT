"""Agent: event currency, session memory, chat face, MCP extras.

The turn loop lives in :class:`src.main.VoiceAgent` (deepagents over the
local model). This package holds the pieces: per-node stream events,
session stores (RAM + JSON-file), the local chat-model face, and the
MCP extra tools (``mcp``: stdio server + client; ``shell`` is the
persistent-shell base for :mod:`src.sandbox.docker`).
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
