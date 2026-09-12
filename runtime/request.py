"""Request/response dataclasses shared by runtime, engine, and server."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VadRequest:
    """Raw audio to segment into speech spans."""

    audio: np.ndarray
    sr: int = 16000
    leg: str = "vad"


@dataclass
class TranscribeRequest:
    """Utterance audio to transcribe."""

    audio: np.ndarray
    sr: int = 16000
    language: str = "en"
    leg: str = "stt"


@dataclass
class ChatRequest:
    """User text for the LLM leg."""

    text: str
    max_tokens: int = 48
    stream: bool = False
    leg: str = "llm"


@dataclass
class SpeakRequest:
    """Text to synthesize."""

    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    leg: str = "tts"


@dataclass
class TurnResult:
    """One full voice turn: user text, reply text, reply audio, timings."""

    text: str = ""
    reply: str = ""
    wav: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sr: int = 24000
    stt_s: float = 0.0
    llm_s: float = 0.0
    tts_s: float = 0.0
    ttfa_s: float = 0.0
    total_s: float = 0.0
    vram_mb: float = 0.0
    segments: Optional[List[Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def timings(self):
        """Timing subset as a plain dict."""
        return {
            "stt_s": self.stt_s,
            "llm_s": self.llm_s,
            "tts_s": self.tts_s,
            "ttfa_s": self.ttfa_s,
            "total_s": self.total_s,
            "vram_mb": self.vram_mb,
        }
