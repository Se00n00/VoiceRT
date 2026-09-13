"""Incremental talk WebSocket: PCM chunks in -> staged events out.

Protocol (one socket = one streaming session, established at connect):
- server -> client: {"event": "ready", "session_id", "sr"} on connect.
- client -> server: binary PCM16 mono frames (default 16 kHz; see config),
  or a one-shot wav file (RIFF bytes, legacy single-turn path).
- client -> server text JSON:
    {"type": "config", "sr": 16000, "end_silence_s": 0.8}
    {"type": "commit"}   end-of-utterance now, run the turn
    {"type": "reset"}    drop buffered audio
    "close"              end the session.
- server -> client staged events, in stage order, on ONE turn:
    {"event": "vad", "speech": bool, "buffer_s": float}   per chunk change
    {"event": "stt", "text": str, "partial": true}   live partials while
        speaking (throttled re-decode of the buffer, best-effort)
    {"event": "stt", "text": str, "partial": false}  final transcript
    {"event": "llm", "token": str}           per generated token
    {"event": "llm", "done": true, "text": str}
    {"event": "tts", "wav_b64": str, "sr": 24000, "sentence": str} per sentence
    {"event": "turn", "text", "reply", "ttfa_s", "total_s", "vram_mb",
     "session_id"}                                           turn summary
- auto-commit: after speech + `end_silence_s` trailing silence, the server
  commits without waiting for {"type": "commit"} (true voice activity).

The engine is imported lazily per connection so importing this module
never pulls torch/weights. The HTTP engine singleton is shared: a second
VoiceEngine would duplicate ~2GB in VRAM and OOM the 4GB card.
"""
import asyncio
import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

CLIENT_SR = 16000
END_SILENCE_S = 0.8
MAX_BUFFER_S = 60.0
# Live partial-STT throttle: decode at most this often, and only once the
# buffer holds this much new audio (a full Whisper pass is ~60ms/3s-audio,
# so per-chunk decoding would just burn GPU for identical prefixes).
PARTIAL_EVERY_S = 1.0
PARTIAL_MIN_S = 0.5


def _get_engine():
    from server.routes import get_engine

    return get_engine()


# One shared instance per process once first WS connects.
_engine = None


def get_ws_engine():
    global _engine
    if _engine is None:
        _engine = _get_engine()
    return _engine


def _pcm16_to_float(data: bytes):
    import numpy as np

    return (np.frombuffer(bytes(data), dtype="<i2").astype("float32")
            / 32768.0)


@router.websocket("/v1/talk")
async def talk(ws: WebSocket):
    await ws.accept()
    try:
        eng = get_ws_engine()
    except Exception as exc:  # weights/torch missing: report, don't hang
        await ws.send_json({"ok": False, "error": f"engine unavailable: {exc}"})
        await ws.close()
        return
    from engine.audio import resample
    from engine.session import new_session_id

    sid = ws.query_params.get("session_id") or new_session_id()
    client_sr = CLIENT_SR
    end_silence_s = END_SILENCE_S
    import numpy as np

    buf = np.zeros(0, dtype=np.float32)
    speech_seen = False
    trailing_sil = 0.0
    last_vad = None
    last_partial_at = 0.0
    last_partial_len = 0
    partial_running = False
    last_partial_text = ""
    await ws.send_json({"event": "ready", "session_id": sid, "sr": 16000})

    async def send_vad(speech):
        nonlocal last_vad
        if speech != last_vad:
            last_vad = speech
            await ws.send_json({"event": "vad", "speech": bool(speech),
                                "buffer_s": len(buf) / 16000.0})

    turning = False

    async def maybe_partial():
        """Live partial transcript of the buffered audio (throttled).

        Re-decodes the whole buffer (one Whisper pass, ~60ms) at most once
        per PARTIAL_EVERY_S and only forwards when the text changed, so the
        client sees words appear while speaking instead of only at commit.
        Never raises — partials are best-effort next to the final decode.
        """
        import time as _time

        nonlocal last_partial_at, last_partial_len, partial_running
        nonlocal last_partial_text
        if partial_running or turning:
            return
        now = _time.monotonic()
        if len(buf) < int(PARTIAL_MIN_S * 16000):
            return
        if (now - last_partial_at < PARTIAL_EVERY_S
                and len(buf) - last_partial_len < int(PARTIAL_MIN_S * 16000)):
            return
        snapshot = np.copy(buf)
        partial_running = True

        async def decode():
            nonlocal last_partial_at, last_partial_len, partial_running
            nonlocal last_partial_text
            try:
                loop = asyncio.get_running_loop()
                r = await loop.run_in_executor(
                    None, lambda: eng.transcribe(snapshot, 16000))
                text = (r.get("text", "") or "").strip()
                if text and text != last_partial_text:
                    last_partial_text = text
                    await ws.send_json({"event": "stt", "text": text,
                                        "partial": True})
                last_partial_at = _time.monotonic()
                last_partial_len = len(snapshot)
            except Exception:
                pass
            finally:
                partial_running = False

        asyncio.create_task(decode())

    async def finalize(buffer):
        """Run one turn over `buffer`, forwarding staged events live."""
        from server.routes import _acquire_or_503, _release
        from runtime.tensor import to_host_numpy

        nonlocal turning
        turning = True
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ticket = _acquire_or_503()

        def emit(kind, payload):
            loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

        def run():
            try:
                r = eng.talk_turn(buffer, 16000, sid, on_event=emit)
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("__done__", r))
            except Exception as exc:  # noqa: BLE001 - forwarded to client
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("__error__", str(exc)[:300]))

        worker = loop.run_in_executor(None, run)
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "__done__":
                    await ws.send_json({
                        "event": "turn", "text": payload["text"],
                        "reply": payload["reply"],
                        "ttfa_s": payload["ttfa_s"],
                        "total_s": payload["total_s"],
                        "vram_mb": payload.get("vram_mb"),
                        "session_id": sid})
                    break
                if kind == "__error__":
                    await ws.send_json({"ok": False, "error": payload})
                    break
                if kind == "stt":
                    await ws.send_json({"event": "stt",
                                        "text": payload["text"],
                                        "partial": False})
                elif kind == "llm":
                    if payload.get("done"):
                        await ws.send_json({"event": "llm", "done": True,
                                            "text": payload.get("text", "")})
                    else:
                        await ws.send_json({"event": "llm",
                                            "token": payload.get("token", "")})
                elif kind == "tts":
                    wav = to_host_numpy(payload["wav"])
                    await ws.send_json({
                        "event": "tts",
                        "wav_b64": base64.b64encode(wav.tobytes()).decode(),
                        "sr": int(payload.get("sr", 24000)),
                        "sentence": payload.get("sentence", "")})
        finally:
            _release(ticket)
            turning = False
            await worker

    def push_pcm(data: bytes):
        """Append one PCM16 frame; returns 'commit' when endpointing fires."""
        nonlocal buf, speech_seen, trailing_sil
        chunk = _pcm16_to_float(data)
        if client_sr != 16000 and len(chunk):
            chunk = resample(chunk, client_sr, 16000)
        buf = np.concatenate([buf, chunk]) if len(buf) else chunk
        if len(buf) / 16000.0 > MAX_BUFFER_S:
            raise ValueError("buffer exceeds 60s; send commit or reset")
        dur = len(chunk) / 16000.0
        speech = eng.vad_active(chunk, 16000) if len(chunk) else False
        if speech:
            speech_seen = True
            trailing_sil = 0.0
        else:
            trailing_sil += dur
        return speech, (speech_seen and trailing_sil >= end_silence_s)

    while True:
        try:
            msg = await ws.receive()
        except WebSocketDisconnect:
            break
        data = msg.get("bytes")
        if data is not None:
            raw = bytes(data)
            if raw[:4] == b"RIFF":
                # Legacy one-shot wav path (unchanged behavior).
                try:
                    from server.routes import _load_wav_sync

                    audio, sr = await asyncio.to_thread(
                        _load_wav_sync, raw)
                    from server.routes import _guard_wav

                    _guard_wav(audio, sr)
                    await finalize(np.asarray(audio, dtype=np.float32)
                                   if sr == 16000
                                   else resample(np.asarray(
                                       audio, dtype=np.float32), sr, 16000))
                except WebSocketDisconnect:
                    break
                except Exception as exc:  # noqa: BLE001
                    try:
                        await ws.send_json({"ok": False,
                                            "error": str(exc)[:300]})
                    except Exception:
                        break
                continue
            try:
                speech, endpoint = await asyncio.to_thread(push_pcm, raw)
                await send_vad(speech)
                await maybe_partial()
                if endpoint:
                    audio, buf = buf, np.zeros(0, dtype=np.float32)
                    speech_seen = False
                    trailing_sil = 0.0
                    last_partial_text = ""
                    last_partial_len = 0
                    await finalize(audio)
            except WebSocketDisconnect:
                break
            except Exception as exc:  # noqa: BLE001
                try:
                    await ws.send_json({"ok": False, "error": str(exc)[:300]})
                except Exception:
                    break
            continue
        text = msg.get("text")
        if text == "close":
            break
        if not text:
            continue
        try:
            import json as _json

            cmd = _json.loads(text)
        except Exception:
            continue
        if not isinstance(cmd, dict):
            continue
        kind = cmd.get("type")
        if kind == "config":
            try:
                client_sr = int(cmd.get("sr", client_sr))
                end_silence_s = float(cmd.get("end_silence_s",
                                              end_silence_s))
            except Exception:
                pass
            await ws.send_json({"event": "ready", "session_id": sid,
                                "sr": 16000})
        elif kind == "reset":
            buf = np.zeros(0, dtype=np.float32)
            speech_seen = False
            trailing_sil = 0.0
            last_partial_text = ""
            last_partial_len = 0
            await ws.send_json({"event": "reset", "buffer_s": 0.0})
        elif kind == "commit":
            if len(buf) == 0:
                await ws.send_json({"ok": False, "error": "empty buffer"})
                continue
            audio, buf = buf, np.zeros(0, dtype=np.float32)
            speech_seen = False
            trailing_sil = 0.0
            last_partial_text = ""
            last_partial_len = 0
            try:
                await finalize(audio)
            except WebSocketDisconnect:
                break
            except Exception as exc:  # noqa: BLE001
                try:
                    await ws.send_json({"ok": False, "error": str(exc)[:300]})
                except Exception:
                    break
