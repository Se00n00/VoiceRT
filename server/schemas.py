"""Request/response schemas for the voice-pipeline HTTP API.

Ported from VOICE/api_server.py (ChatReq/SpeakReq) plus explicit response
models for every leg so tests and clients have a stable contract.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_tokens: int = Field(default=48, ge=1, le=256)
    stream: bool = False
    # Multi-turn memory: client UUID (hex). Absent -> stateless turn.
    session_id: Optional[str] = Field(default=None, max_length=64)
    reset: bool = False  # clear this session's history before the turn


class SpeakReq(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class VadResp(BaseModel):
    segments: List[List[float]] = Field(default_factory=list)
    audio_dur_s: float = 0.0
    n_segments: int = 0


class TranscribeResp(BaseModel):
    text: str = ""
    rtf: float = 0.0
    ttfs: Optional[float] = None
    # engine reports `dur`; older STT servers reported `audio_dur_s`.
    # Both accepted; `dur` is canonical here.
    dur: float = 0.0
    audio_dur_s: Optional[float] = None


class ChatResp(BaseModel):
    text: str = ""
    ttft_s: float = 0.0
    tps: float = 0.0
    session_id: Optional[str] = None


class VoiceResp(BaseModel):
    text: str = ""
    reply: str = ""
    wav_b64: str = ""
    ttfa_s: float = 0.0
    total_s: float = 0.0
    vram_mb: Optional[float] = None
    session_id: Optional[str] = None
