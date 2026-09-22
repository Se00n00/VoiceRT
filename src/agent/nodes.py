"""LangGraph nodes: one async function per pipeline stage.

Each node takes the turn state, returns a state update, and streams
:class:`AgentEvent` through the LangGraph ``StreamWriter``. Models arrive
via ``functools.partial`` binding in :mod:`src.agent.graph`, so nodes stay
directly unit-testable with fakes (pass an explicit ``writer`` such as
``list.append`` when calling outside a graph run).
"""
import time

import numpy as np

from src.agent.events import AgentEvent

__all__ = [
    "vad_node",
    "stt_node",
    "respond_node",
    "silence_node",
    "route_after_vad",
    "turn_summary",
]

_TRIM_PAD_S = 0.15


def _writer(provided=None):
    """StreamWriter in-graph, explicit callable in tests, null otherwise."""
    if provided is not None:
        return provided
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return lambda event: None


def _timed(node_s, name, dt):
    merged = dict(node_s or {})
    merged[name] = merged.get(name, 0.0) + dt
    return merged


def _trim(audio, sr, segs, pad_s=_TRIM_PAD_S):
    wav = np.asarray(audio, dtype=np.float32).ravel()
    dur = len(wav) / float(sr)
    s0 = max(0.0, float(segs[0][0]) - pad_s)
    s1 = min(dur, float(segs[-1][1]) + pad_s)
    if s1 - s0 < 0.05 or (s0 <= 0.0 and s1 >= dur):
        return audio
    return wav[int(s0 * sr):int(s1 * sr)]


def turn_summary(state, *, total, first_audio_at) -> AgentEvent:
    """Turn-summary event from a (near-)final state."""
    try:
        from src.models.runtime import max_allocated_mb

        vram_mb = float(max_allocated_mb())
    except Exception:
        vram_mb = None
    return AgentEvent(node="turn", kind="summary", data={
        "text": state.get("text", ""),
        "reply": state.get("reply", ""),
        "segments": state.get("segments", []),
        "node_s": dict(state.get("node_s") or {}),
        "ttfa_s": first_audio_at if first_audio_at is not None else total,
        "total_s": total,
        "vram_mb": vram_mb,
        "session_id": state.get("sid"),
    })


async def vad_node(state, *, vad, trim_pad_s=_TRIM_PAD_S, writer=None):
    """Speech spans (+ STT-window trim). Silence passes through empty."""
    w = _writer(writer)
    sr = state.get("sr", 16000)
    t0 = time.perf_counter()
    try:
        res = await vad.segments(state["audio"], sr)
    finally:
        dt = time.perf_counter() - t0
    segments = [[float(a), float(b)] for a, b in res.segments]
    audio = (state["audio"] if not segments
             else _trim(state["audio"], sr, segments, trim_pad_s))
    w(AgentEvent(node="vad", kind="segments", data={
        "segments": segments,
        "audio_dur_s": float(res.audio_dur_s),
        "speech_s": float(res.speech_s),
    }))
    return {"segments": segments, "audio": audio,
            "node_s": _timed(state.get("node_s"), "vad", dt)}


def route_after_vad(state) -> str:
    """Silence short-circuits the turn; speech goes to STT."""
    return "stt" if state.get("segments") else "silence"


async def silence_node(state, *, writer=None):
    """Silent turn: stt-empty event + summary, no STT/LLM/TTS burn."""
    w = _writer(writer)
    total = time.perf_counter() - state.get("t0", time.perf_counter())
    w(AgentEvent(node="stt", kind="text",
                 data={"text": "", "silent": True}))
    w(turn_summary({**state, "text": "", "reply": ""},
                   total=total, first_audio_at=None))
    return {"silent": True, "text": "", "reply": ""}


async def stt_node(state, *, stt, writer=None):
    """Transcribe the (VAD-trimmed) utterance."""
    w = _writer(writer)
    sr = state.get("sr", 16000)
    t0 = time.perf_counter()
    try:
        res = await stt.transcribe(state["audio"], sr)
    finally:
        dt = time.perf_counter() - t0
    text = str(res.text)
    w(AgentEvent(node="stt", kind="text", data={
        "text": text, "rtf": float(res.rtf),
        "ttfs": float(res.ttfs), "dur_s": float(res.dur_s),
    }))
    return {"text": text, "node_s": _timed(state.get("node_s"), "stt", dt)}


async def respond_node(state, *, llm, tts, sessions=None, writer=None):
    """Stream reply tokens, synthesize each sentence mid-decode (TTFA),
    remember the turn, then emit llm-done + the turn summary."""
    from engine import SentenceSplitter

    w = _writer(writer)
    t0 = state.get("t0", time.perf_counter())
    first_audio_at = None
    node_s = dict(state.get("node_s") or {})
    reply_ids = list(state.get("reply_ids") or [])

    t_llm = time.perf_counter()
    messages = llm.messages(state.get("text", ""),
                            state.get("history") or [])
    pending_done: AgentEvent | None = None
    try:
        from src.models.llm import split_thinking

        splitter = SentenceSplitter()
        # Thinking gate: hold pieces out of the TTS splitter while inside
        # a <think> block, so thoughts are shown but never spoken. Tokens
        # themselves still stream (visible as thinking in the UI).
        pre_buf = ""
        pre_n = 0
        think_emitted = False
        THINK_CAP = 80

        async def _speak_sentence(sent):
            nonlocal first_audio_at
            t_tts = time.perf_counter()
            try:
                out = await tts.speak(sent)
            finally:
                node_s["tts"] = (node_s.get("tts", 0.0)
                                 + time.perf_counter() - t_tts)
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - t0
            w(AgentEvent(node="tts", kind="audio", data={
                "wav": np.asarray(out.wav, dtype=np.float32),
                "sr": int(out.sample_rate),
                "sentence": str(out.sentence or sent),
                "synth_s": float(out.synth_s),
            }))

        async for tok in llm.stream(messages):
            reply_ids.append(int(tok.token_id))
            w(AgentEvent(node="llm", kind="token", data={
                "token": tok.piece, "first": bool(tok.first)}))
            piece = tok.piece or ""
            if not think_emitted:
                pre_buf += piece
                pre_n += 1
                low = pre_buf.lower()
                if "</think>" in low:
                    thinking, tail = split_thinking(pre_buf)
                    if thinking:
                        w(AgentEvent(node="llm", kind="thinking",
                                     data={"text": thinking[:2000]}))
                    think_emitted = True
                    for sent in splitter.push(tail):
                        await _speak_sentence(sent)
                    pre_buf = ""
                elif "<think>" not in low and (
                        pre_n >= THINK_CAP or "." in pre_buf or "?" in pre_buf
                        or "!" in pre_buf or "\n" in pre_buf):
                    # Answer flowing with no think block in sight: normal
                    # streaming path. (A think tag opening later still gets
                    # caught by the final split, just not pre-gated.)
                    think_emitted = True
                    for sent in splitter.push(pre_buf):
                        await _speak_sentence(sent)
                    pre_buf = ""
                elif pre_n >= THINK_CAP:
                    # Unclosed think running long: emit thinking so far, then
                    # resume streaming (final split will handle remainder).
                    thinking, _ = split_thinking(pre_buf)
                    if thinking:
                        w(AgentEvent(node="llm", kind="thinking",
                                     data={"text": thinking[:2000]}))
                    think_emitted = True
                    pre_buf = ""
                continue
            for sent in splitter.push(piece):
                await _speak_sentence(sent)
        reply = await llm.decode(reply_ids)
        thinking, answer = split_thinking(reply)
        if thinking and not think_emitted:
            w(AgentEvent(node="llm", kind="thinking",
                         data={"text": thinking[:2000]}))
        if pre_buf:
            # Stream ended while holding (unclosed think or punctuation-free
            # tail): speak only the answer part, never raw thoughts.
            _, tail = split_thinking(pre_buf)
            for sent in splitter.push(tail):
                await _speak_sentence(sent)
            pre_buf = ""
        reply = answer
        pending_done = AgentEvent(node="llm", kind="done",
                                  data={"text": reply})
        tail = splitter.flush()
        if tail and tail.strip():
            t_tts = time.perf_counter()
            try:
                out = await tts.speak(tail)
            finally:
                node_s["tts"] = (node_s.get("tts", 0.0)
                                 + time.perf_counter() - t_tts)
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - t0
            w(AgentEvent(node="tts", kind="audio", data={
                "wav": np.asarray(out.wav, dtype=np.float32),
                "sr": int(out.sample_rate),
                "sentence": str(out.sentence or tail),
                "synth_s": float(out.synth_s),
            }))
    finally:
        node_s["llm"] = node_s.get("llm", 0.0) + (
            time.perf_counter() - t_llm)
    if pending_done is not None:
        w(pending_done)
    if sessions is not None and state.get("remember") and (state.get("text") or reply):
        sessions.remember_turn(state.get("sid"), state.get("text"), reply)
    total = time.perf_counter() - t0
    w(turn_summary({**state, "reply": reply, "reply_ids": reply_ids,
                    "node_s": node_s},
                   total=total, first_audio_at=first_audio_at))
    return {"reply_ids": reply_ids, "reply": reply, "node_s": node_s,
            "first_audio_at": first_audio_at}
