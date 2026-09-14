"""Per-node stream events: the single currency of the agent loop.

Every pipeline node (vad/stt/llm/tts) reports progress by yielding
:class:`AgentEvent`; the ``VoiceAgent.__call__`` loop only forwards them
(transport layers serialize). ``node`` is one of :data:`NODES` (or
``"turn"`` for the loop summary); ``kind`` names the payload shape.
"""
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NODES", "AgentEvent"]

NODES = ("vad", "stt", "llm", "tts")


@dataclass(frozen=True)
class AgentEvent:
    node: str = "vad"   # vad | stt | llm | tts | turn
    kind: str = ""      # segments | text | token | done | audio | summary | error
    data: dict = field(default_factory=dict)
    t_s: float = field(default_factory=time.perf_counter)

    def as_dict(self) -> dict:
        """JSON-safe view (ndarrays are the caller's job to encode)."""
        data: dict[str, Any] = {}
        for k, v in (self.data or {}).items():
            try:
                import numpy as _np

                if isinstance(v, _np.ndarray):
                    v = {"dtype": str(v.dtype), "shape": list(v.shape),
                         "n": int(v.size)}
            except Exception:
                pass
            data[k] = v
        return {"node": self.node, "kind": self.kind, "data": data,
                "t_s": float(self.t_s)}
