"""Voice agent package: VoiceAgent shell and the VAD/STT/LLM/TTS cascade.

The voice entry points live here, over the long-running harness and the
model legs. Import ``VoiceAgent``/``VoiceAgentConfig`` from this package;
callers never reach into ``cascade.py`` directly.
"""

from src.agent.voice_agent.agent import VoiceAgent, VoiceAgentConfig
from src.agent.voice_agent.cascade import CascadeConfig, CascadePipeline

__all__ = [
    "VoiceAgent",
    "VoiceAgentConfig",
    "CascadeConfig",
    "CascadePipeline",
]
