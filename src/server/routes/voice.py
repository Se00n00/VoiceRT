"""Voice legs: one-shot STT, live STT streaming, TTS."""

import asyncio
import base64
import json

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..state import _wav_b64, get_agent

router = APIRouter()


@router.post("/term/stt")
async def term_stt(payload: dict):
    """Mic PCM16 mono -> transcript (VAD gate + Whisper)."""
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


@router.websocket("/term/stt-stream")
async def term_stt_stream(ws: WebSocket):
    """Live voice pipeline: binary PCM16 mono frames in, transcripts out.

    The client streams raw int16 frames (any chunk size; 16kHz mono is
    the contract, other rates are resampled with numpy). Every
    PARTIAL_EVERY_S the server VAD-gates the trailing window (up to
    PARTIAL_WINDOW_S) and, when speech is present, transcribes it and
    emits a partial. ``{"type": "stop"}`` ends the utterance: one final
    full transcribe, a ``final`` event, then close. Silence anywhere
    yields no text, never an error.
    """
    PARTIAL_EVERY_S = 2.0
    PARTIAL_WINDOW_S = 30.0
    await ws.accept()
    sr = 16000
    buf = np.zeros(0, dtype=np.float32)
    dirty = {"n": 0}
    closed = {"done": False}
    running = {"partial": False}

    async def maybe_partial(force: bool = False) -> None:
        if running["partial"] or closed["done"]:
            return
        if buf.size < 1600:
            return
        if not force and dirty["n"] == 0:
            return
        dirty["n"] = 0
        running["partial"] = True
        try:
            win_n = int(min(len(buf), PARTIAL_WINDOW_S * sr))
            if win_n < 1600:
                return
            win = buf[-win_n:].copy()
            agent = get_agent()
            try:
                segs = await agent.vad.segments(win, sr)
            except Exception:
                return
            if not segs.segments:
                return
            try:
                res = await agent.stt.transcribe(win, sr)
            except Exception:
                return
            text = (res.text or "").strip()
            if not text:
                return
            try:
                await ws.send_json({"event": "partial", "text": text})
            except Exception:
                closed["done"] = True
        finally:
            running["partial"] = False

    async def partial_loop() -> None:
        try:
            while not closed["done"]:
                await asyncio.sleep(PARTIAL_EVERY_S)
                await maybe_partial()
        except asyncio.CancelledError:
            pass

    async def finalize() -> None:
        closed["done"] = True
        try:
            if buf.size >= 1600:
                agent = get_agent()
                try:
                    segs = await agent.vad.segments(buf, sr)
                except Exception:
                    segs = None
                if segs is not None and segs.segments:
                    try:
                        res = await agent.stt.transcribe(buf, sr)
                    except Exception as exc:
                        try:
                            await ws.send_json({"event": "error",
                                                "message": f"STT failed: {exc}"[:200]})
                        except Exception:
                            pass
                        return
                    text = (res.text or "").strip()
                    try:
                        await ws.send_json({"event": "final", "text": text})
                    except Exception:
                        pass
                    return
            try:
                await ws.send_json({"event": "final", "text": ""})
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    loop_task = asyncio.create_task(partial_loop())
    try:
        while not closed["done"]:
            try:
                msg = await ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                continue
            data = msg.get("bytes")
            if data is not None:
                try:
                    chunk = (np.frombuffer(bytes(data), dtype=np.int16)
                               .astype("float32") / 32768.0)
                except Exception:
                    continue
                if chunk.size:
                    buf = np.concatenate([buf, chunk]) if buf.size else chunk
                    dirty["n"] += chunk.size
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                obj = json.loads(text) if isinstance(text, str) else {}
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            kind = str(obj.get("type") or "")
            if kind == "hello":
                try:
                    sr = max(8000, min(48000, int(obj.get("sr", 16000))))
                except Exception:
                    sr = 16000
            elif kind == "stop":
                break
    finally:
        closed["done"] = True
        try:
            loop_task.cancel()
        except Exception:
            pass
        await finalize()


@router.post("/term/say")
async def term_say(payload: dict):
    """Text -> Kokoro wav (float32 b64) for Ink playback + viz."""
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
