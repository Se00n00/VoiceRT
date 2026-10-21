"""Agent: event currency, session memory, chat face, MCP extras.

The turn loop lives in :class:`src.agent.long_running.agent.LongRunningAgent`;
:class:`src.agent.voice_agent.agent.VoiceAgent` adds the voice cascade.
This package holds the shared pieces: per-node stream events, session stores (RAM +
JSON-file), the local chat-model face, and the MCP extra tools (``mcp``:
stdio server + client; ``shell`` is the persistent-shell base for
:mod:`src.agent.sandbox.docker`).
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
