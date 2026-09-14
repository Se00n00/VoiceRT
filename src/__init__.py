"""voice-agent v2: clean-room layout over the proven legs.

``src`` holds the new structure; ``engine/`` (audio/session/streaming
helpers), ``server.py`` (the single API server) and the original
``models/`` package live at the root and are reused by import — never
copied.

- ``src.models``: one clean async class per leg, dataclass configs, no YAML.
- ``src.agent``: LangGraph turn graph (state, nodes, builder) over
  per-node stream events.
- ``src.main``: :class:`VoiceAgent` — VAD -> STT -> LLM -> TTS loop.
"""
