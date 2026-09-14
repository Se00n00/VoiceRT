"""Model legs: clean async classes with dataclass configs (no YAML).

Each ``<leg>.py`` module owns one ``<Leg>Config`` frozen dataclass and one
``<Leg>Model`` class with async methods. Heavy backends (the proven legs in
top-level ``models/``) are imported lazily and driven in worker threads, so
importing this package never loads weights.
"""
from src.models.llm import LlmConfig, LlmModel, LlmResult, LlmToken
from src.models.stt import SttConfig, SttModel, SttResult
from src.models.tts import TtsAudio, TtsConfig, TtsModel
from src.models.vad import VadConfig, VadModel, VadSegments

__all__ = [
    "VadConfig", "VadModel", "VadSegments",
    "SttConfig", "SttModel", "SttResult",
    "LlmConfig", "LlmModel", "LlmResult", "LlmToken",
    "TtsConfig", "TtsModel", "TtsAudio",
]
