"""Turn state: the single dict flowing through the LangGraph turn graph.

``total=False`` so nodes return only the keys they set; the graph merges
them. Lists are always replaced wholesale by the node that owns them
(the graph is linear, so no fan-in reducers are needed).
"""
from typing import Any
from typing import TypedDict

__all__ = ["AgentState"]


class AgentState(TypedDict, total=False):
    audio: Any             # utterance waveform (VAD trims it in place)
    sr: int                # sample rate
    sid: str               # effective session id (fresh when stateless)
    remember: bool         # store this turn (only when client sent an id)
    history: list          # session messages for the LLM prompt
    segments: list         # VAD spans [[start_s, end_s]]
    text: str              # STT transcript
    reply_ids: list        # generated token ids
    reply: str             # decoded reply
    node_s: dict           # per-node seconds
    silent: bool           # VAD gate found no speech
    t0: float              # turn start (perf_counter)
    first_audio_at: float | None  # seconds to first TTS chunk
