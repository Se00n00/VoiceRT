"""voice-agent v2: clean-room layout over the proven legs.

- ``src.models``: one clean async class per leg, dataclass configs, no YAML.
- ``src.agent``: the agent surface — ``VoiceAgent``, the long-running
  harness, prompts, sandbox, memory, MCP tools and storage.
- ``src.server``: the FastAPI backend on ``:8004``.
- ``src.tauri``: the Tauri + SolidJS desktop client.
"""
