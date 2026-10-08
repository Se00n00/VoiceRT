"""Full-duplex voice calls over WebRTC (barge-in capable).

Signaling (localhost only — no STUN/TURN)::

    POST /talk/offer {"sdp": <offer>, "sid": <session or "">}
      -> {"sdp": <answer>, "sid": <session>}
    audio: mic Opus in, synthesized reply Opus out (one audio transceiver)
    datachannel "voice" (client-created):
      server -> client: {"type": "state", "state": "listening|thinking|speaking"}
      server -> client: {"type": "partial", "text": ...}   (utterance so far)
      server -> client: {"type": "transcript", "role": "user"|"agent", "text": ...}
      server -> client: {"type": "barge-in"}

Pipeline per utterance: energy-gated VAD -> Whisper -> Qwen generate
(cancellable task) -> sentence-chunked Kokoro -> outbound track. Speech
while responding cancels the LLM task, drains the TTS queue and drops
back to listening. One active call per process (4GB box).

Sample rates: mic arrives at 48kHz (WebRTC), STT/VAD run at 16kHz,
Kokoro renders at 24kHz — resampled with numpy at the boundaries.
"""

import asyncio
import json
import re
from fractions import Fraction

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..state import get_agent, note_turn_end, note_turn_start, qwen_leg

router = APIRouter()

_FRAME_S48 = 960
_SR_IN = 48000
_SR_STT = 16000
_SR_TTS = 24000

_VAD_START = 0.02
_VAD_BARGE = 0.03
_BARGE_HOLD_S = 0.3
_END_SILENCE_S = 0.8
_MIN_UTTER_S = 0.4
_MAX_UTTER_S = 30.0
_HISTORY_TURNS = 20

_active_call: dict = {}


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    if sr_in == sr_out or x.size == 0:
        return x
    n_out = max(1, int(round(x.size * sr_out / sr_in)))
    old_idx = np.arange(x.size)
    new_idx = np.linspace(0, x.size - 1, n_out)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def _split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


class TtsOutboundTrack(MediaStreamTrack):
    """Server -> client reply audio (48kHz mono). Fed sentence by sentence."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ts = 0

    def push_pcm24(self, wav) -> None:
        try:
            f48 = _resample(np.asarray(wav, dtype=np.float32), _SR_TTS, _SR_IN)
            i16 = (np.clip(f48, -1.0, 1.0) * 32767).astype(np.int16)
            for i in range(0, max(_FRAME_S48, len(i16)), _FRAME_S48):
                chunk = i16[i:i + _FRAME_S48]
                if len(chunk) < _FRAME_S48:
                    pad = np.zeros(_FRAME_S48 - len(chunk), dtype=np.int16)
                    chunk = np.concatenate([chunk, pad])
                try:
                    self._queue.put_nowait(chunk.tobytes())
                except Exception:
                    return
        except Exception:
            pass

    def drain(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass

    async def recv(self):
        try:
            data = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            samples = np.frombuffer(data, dtype=np.int16)
            if samples.size != _FRAME_S48:
                fixed = np.zeros(_FRAME_S48, dtype=np.int16)
                fixed[:min(_FRAME_S48, samples.size)] = samples[:_FRAME_S48]
                samples = fixed
        except asyncio.TimeoutError:
            samples = np.zeros(_FRAME_S48, dtype=np.int16)
        frame = av.AudioFrame(format="s16", layout="mono", samples=_FRAME_S48)
        frame.pts = self._ts
        frame.sample_rate = _SR_IN
        frame.time_base = Fraction(1, _SR_IN)
        frame.planes[0].update(samples.tobytes())
        self._ts += _FRAME_S48
        return frame


class TalkCall:
    def __init__(self, pc: RTCPeerConnection, sid: str):
        self.pc = pc
        self.sid = sid
        self.state = "listening"
        self.outbound = TtsOutboundTrack()
        self.channel = None
        self.buf: list = []
        self.buf_s = 0.0
        self.speaking = False
        self.quiet_s = 0.0
        self.barge_s = 0.0
        self.pipeline = None
        self.reader = None
        self.done = False

    def send_dc(self, obj: dict) -> None:
        try:
            if self.channel is not None and getattr(self.channel, "readyState", "") == "open":
                self.channel.send(json.dumps(obj))
        except Exception:
            pass

    def set_state(self, s: str) -> None:
        self.state = s
        self.send_dc({"type": "state", "state": s})

    def history_messages(self, agent) -> list:
        try:
            sessions = getattr(agent, "sessions", None)
            if sessions is None or not self.sid:
                return []
            hist = sessions.history(self.sid) or []
            msgs = [{"role": str(m.get("role", "user") or "user"),
                     "content": str(m.get("content", "") or "")}
                    for m in hist if isinstance(m, dict)
                    and str(m.get("content", "") or "").strip()
                    and str(m.get("role", "user") or "user") in ("user", "assistant")]
            return msgs[-(_HISTORY_TURNS * 2):]
        except Exception:
            return []

    def store_turn(self, agent, role: str, content: str) -> None:
        try:
            sessions = getattr(agent, "sessions", None)
            if sessions is not None and self.sid and str(content or "").strip():
                sessions.append(self.sid, role, str(content))
        except Exception:
            pass

    async def cancel_pipeline(self) -> None:
        task = self.pipeline
        self.pipeline = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def run_utterance(self, wav16: np.ndarray) -> None:
        t0 = note_turn_start()
        try:
            agent = get_agent()
            try:
                segs = await agent.vad.segments(wav16, _SR_STT)
            except Exception:
                return
            if not segs.segments:
                return
            try:
                res = await agent.stt.transcribe(wav16, _SR_STT)
            except Exception:
                return
            text = (getattr(res, "text", "") or "").strip()
            if not text:
                return
            self.send_dc({"type": "transcript", "role": "user", "text": text})
            self.store_turn(agent, "user", text)
            llm = qwen_leg(agent)
            if llm is None:
                return
            messages = self.history_messages(agent) + [{"role": "user", "content": text}]
            self.set_state("thinking")
            gen_task = asyncio.ensure_future(llm.generate(messages, max_tokens=256))
            self.pipeline = gen_task
            try:
                res = await gen_task
            except asyncio.CancelledError:
                return
            finally:
                if self.pipeline is gen_task:
                    self.pipeline = None
            reply = str(getattr(res, "text", "") or "").strip()
            if not reply or self.done:
                return
            self.store_turn(agent, "agent", reply)
            self.send_dc({"type": "transcript", "role": "agent", "text": reply})
            self.set_state("speaking")
            tts = getattr(agent, "tts", None)
            if tts is None:
                return
            try:
                stream = tts.speak_stream(_split_sentences(reply) or [reply])
                async for out in stream:
                    wav = getattr(out, "wav", None)
                    if wav is None:
                        continue
                    self.outbound.push_pcm24(wav)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        finally:
            note_turn_end(t0)
            if not self.done and self.state != "listening":
                self.set_state("listening")

    def launch_utterance(self, wav16: np.ndarray) -> None:
        if self.pipeline is not None and not self.pipeline.done():
            return
        if self.done:
            return
        self.pipeline = asyncio.ensure_future(self.run_utterance(wav16))

    async def ingest(self, f32_16k: np.ndarray) -> None:
        if self.done:
            return
        try:
            rms = float(np.sqrt(max(1e-9, float(np.mean(f32_16k ** 2)))))
        except Exception:
            return
        dt = len(f32_16k) / _SR_STT
        if self.state == "speaking":
            if rms >= _VAD_BARGE:
                self.barge_s += dt
                if self.barge_s >= _BARGE_HOLD_S:
                    self.barge_s = 0.0
                    self.buf = []
                    self.buf_s = 0.0
                    self.speaking = False
                    self.quiet_s = 0.0
                    await self.cancel_pipeline()
                    self.outbound.drain()
                    self.send_dc({"type": "barge-in"})
                    self.set_state("listening")
            else:
                self.barge_s = 0.0
            return
        if self.pipeline is not None and not self.pipeline.done():
            return
        if rms >= _VAD_START:
            if not self.speaking:
                self.speaking = True
                self.quiet_s = 0.0
            self.buf.append(np.asarray(f32_16k, dtype=np.float32))
            self.buf_s += dt
            if self.buf_s >= _MAX_UTTER_S:
                await self._finish_utterance()
        elif self.speaking:
            self.buf.append(np.asarray(f32_16k, dtype=np.float32))
            self.buf_s += dt
            self.quiet_s += dt
            if self.quiet_s >= _END_SILENCE_S:
                await self._finish_utterance()

    async def _finish_utterance(self) -> None:
        self.speaking = False
        self.quiet_s = 0.0
        if self.buf_s < _MIN_UTTER_S or not self.buf:
            self.buf = []
            self.buf_s = 0.0
            return
        wav = np.concatenate(self.buf) if len(self.buf) > 1 else self.buf[0]
        self.buf = []
        self.buf_s = 0.0
        self.launch_utterance(np.asarray(wav, dtype=np.float32))

    async def close(self) -> None:
        if self.done:
            return
        self.done = True
        await self.cancel_pipeline()
        if self.reader is not None and not self.reader.done():
            self.reader.cancel()
        try:
            await self.pc.close()
        except Exception:
            pass


async def _wait_gathering(pc: RTCPeerConnection, timeout: float = 5.0) -> None:
    if pc.iceGatheringState == "complete":
        return
    ev = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _check():
        if pc.iceGatheringState == "complete":
            try:
                ev.set()
            except Exception:
                pass

    try:
        await asyncio.wait_for(ev.wait(), timeout)
    except asyncio.TimeoutError:
        pass


@router.post("/talk/offer")
async def talk_offer(payload: dict):
    """WebRTC offer/answer handshake for a duplex voice call."""
    from aiortc import RTCSessionDescription

    if _active_call.get("call") is not None:
        return JSONResponse(status_code=409, content={"error": "call already active"})
    try:
        sdp = str((payload or {}).get("sdp", "") or "")
        if not sdp:
            return JSONResponse(status_code=400, content={"error": "missing sdp offer"})
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad payload"})
    import time as _time

    sid = str((payload or {}).get("sid", "") or "").strip() or ("call-%d" % int(_time.time()))
    pc = RTCPeerConnection()
    call = TalkCall(pc, sid)

    @pc.on("datachannel")
    def _on_dc(channel):
        call.channel = channel

    @pc.on("track")
    def _on_track(track):
        if track.kind != "audio":
            return

        async def _read():
            while not call.done:
                try:
                    frame = await track.recv()
                except Exception:
                    break
                try:
                    # to_ndarray is one interleaved plane: reshape to
                    # (samples, channels) before the mono mixdown.
                    arr = frame.to_ndarray()
                    try:
                        nch = len(frame.layout.channels)
                    except Exception:
                        nch = 1
                    flat = np.asarray(arr).ravel().astype(np.float32)
                    if nch > 1 and flat.size % nch == 0:
                        mono = flat.reshape(-1, nch).mean(axis=1)
                    else:
                        mono = flat
                    f16 = _resample(mono / 32768.0, _SR_IN, _SR_STT)
                    await call.ingest(f16)
                except Exception:
                    continue

        call.reader = asyncio.ensure_future(_read())

    @pc.on("connectionstatechange")
    async def _on_state():
        if pc.connectionState in ("failed", "closed"):
            if _active_call.get("call") is call:
                _active_call.pop("call", None)
            await call.close()

    pc.addTrack(call.outbound)
    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_gathering(pc)
    except Exception as exc:
        try:
            await pc.close()
        except Exception:
            pass
        return JSONResponse(status_code=400, content={"error": f"webrtc failed: {exc}"[:200]})
    _active_call["call"] = call
    call.set_state("listening")
    return {"sdp": pc.localDescription.sdp, "sid": sid}
