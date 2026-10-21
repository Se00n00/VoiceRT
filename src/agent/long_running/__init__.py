"""Long-running agent package: harness, specialists, middleware, prompts.

Everything the text/agentic turn loop owns lives here. Shared layers
stay outside on purpose: chat models (``src.models``), execution
tools/MCP (``src.agent.tool_router``, ``src.agent.mcp``), and session
events (``src.agent.events``). ``src.agent.voice_agent.agent.VoiceAgent``
subclasses :class:`LongRunningAgent` and adds the voice cascade.
"""

from src.agent.long_running.agent import LongRunningAgent, LongRunningConfig
from src.agent.long_running.messages import lc_messages
from src.agent.long_running.speech import cap_words, speak_head, vram_mb

__all__ = [
    "LongRunningAgent",
    "LongRunningConfig",
    "cap_words",
    "lc_messages",
    "speak_head",
    "vram_mb",
]
