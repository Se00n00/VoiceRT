"""Voice service: one-shot STT and TTS for the voice routes.

Contract: plain dict payloads consumed by src.server.routes.voice.
Dependencies: numpy, the agent's VAD/STT/TTS legs.
Used by: src.server.routes.voice.
"""

import base64

import numpy as np

from ..state import _wav_b64, get_agent


async def transcribe(payload: dict) -> dict:
    """Mic PCM16 mono -> transcript (VAD gate + Whisper).

    ..
    """
    try:
        raw = base64.b64decode((payload or {}).get("pcm_b64", ""))
    except Exception:
        return {"kind": "error", "message": "bad pcm_b64"}
    sr = int((payload or {}).get("sr", 16000))
    wav = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
    if wav.size < 1600:
        return {"kind": "error", "message": "nothing recorded"}
    agent = get_agent()
    try:
        segs = await agent.vad.segments(wav, sr)
    except Exception as exc:
        return {"kind": "error", "message": f"VAD failed: {exc}"[:200]}
    if not segs.segments:
        return {"kind": "empty", "message": "silence"}
    try:
        res = await agent.stt.transcribe(wav, sr)
    except Exception as exc:
        return {"kind": "error", "message": f"STT failed: {exc}"[:200]}
    if not res.text.strip():
        return {"kind": "empty", "message": "STT empty"}
    return {"kind": "text", "text": res.text}


async def synthesize(payload: dict) -> dict:
    """Text -> Kokoro wav (float32 b64) for Ink playback + viz.

    ..
    """
    text = str((payload or {}).get("text", "") or "")[:500]
    if not text.strip():
        return {"kind": "error", "message": "empty text"}
    agent = get_agent()
    try:
        out = await agent.tts.speak(text)
    except Exception as exc:
        return {"kind": "error", "message": f"TTS failed: {exc}"[:200]}
    return {"kind": "audio", "wav_b64": _wav_b64(out.wav),
            "sr": int(out.sample_rate)}
