"""VoiceAgent: ONE autonomous agent for voice, terminal, and everything else.

Device placement (two brains: Qwen3-0.6B fused front on GPU/CPU +
Bonsai-27B sidecar worker):

- LLM  Qwen3-0.6B fused Triton (``backend="qwen"``) or Bonsai 27B
  llama.cpp sidecar (``backend="bonsai"``) for agentic work.
- VAD  Silero ONNX (CPU, RTF ~0.01).
- STT  Whisper-base pure-torch (GPU when CUDA is up, CPU fallback otherwise).
- TTS  Kokoro-82M + torch post (GPU when CUDA is up, CPU fallback otherwise).

Pipeline: speech -> VAD spans -> STT text -> deep-agent turn (tools +
JSON session memory) -> TTS. Text turns go through :meth:`run_text`
(TUI, bridge, CLI, evals); voice turns through :meth:`__call__`
(``server.py`` ``/talk``). If a turn is already running for a session,
new input is queued and injected as follow-up message(s) once the
active turn drains.
"""

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.cascade import CascadeConfig, CascadePipeline
from src.long_running import (
    LongRunningAgent,
    LongRunningConfig,
    _lc_messages,
)
from src.models.llm import split_thinking

__all__ = ["VoiceAgentConfig", "VoiceAgent"]

class VoiceAgentConfig(BaseModel):
    """Two-brain shell config: cascade + harness sub-configs.

    Flat kwargs are gone: pass cascade=/harness= configs (step 3 adds
    front/worker routing fields here).
    ..
    """

    model_config = ConfigDict(frozen=True)

    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    harness: LongRunningConfig = Field(default_factory=LongRunningConfig)
    fast_voice: bool = False


class VoiceAgent(LongRunningAgent):
    """Text + voice entry points. Two-brain composition lands here.

    Behavioral shell over the harness + cascade: every attribute and
    method external callers use keeps working unchanged.
    ..
    """

    def __init__(self, config: VoiceAgentConfig | None = None) -> None:
        """Build the harness, then the cascade on it. Warms nothing.

        ..
        """
        cfg = config or VoiceAgentConfig()
        super().__init__(cfg.harness)
        self.cascade = CascadePipeline(cfg.cascade, self)
        self.tts = self.cascade.tts
        self.fast_voice = bool(cfg.fast_voice)

    def __call__(self, audio_or_text, sr: int = 16000, session_id: str | None = None):
        """Text or voice dispatch (both transports share one object).

        - ``await agent("compute it", session_id="t")`` — plain text turn,
          returns the final reply string (CLI-style / unit tests).
        - ``async for ev in agent(wav, sr, sid)`` — voice turn, yields
          vad/stt/llm/tts/term events + summary (``server.py`` ``/talk``).
        """
        if isinstance(audio_or_text, str):
            return self._text_reply(str(audio_or_text), session_id=session_id)
        if (
            bool(self.fast_voice)
            and self.delegate_cfg is None
        ):
            return self._voice_fast(audio_or_text, sr=sr, session_id=session_id)
        return self._voice_turns(audio_or_text, sr=sr, session_id=session_id)

    async def _text_reply(self, text: str, session_id: str | None = None) -> str:
        """One text turn, reply string only (no audio synthesis).

        With delegation on, the front brain routes first: a chat route
        answers here and never builds the worker graph at all, which is the
        whole point of the split on a text path that otherwise pays for
        agent setup on "hi".
        """
        hist = self.sessions.history(session_id) if session_id else []
        if self.front_llm is not None and self.delegate_cfg is not None:
            route = await self._front_decide(str(text), hist)
            if getattr(self.delegate_cfg, "emit_route", True):
                print(
                    f"[route] {'front' if route.kind == 'chat' else 'worker'}"
                    f" ({route.reason})",
                    flush=True,
                )
            if route.kind == "chat" and str(route.text or "").strip():
                reply = str(route.text).strip()
                if session_id:
                    try:
                        self.sessions.remember_turn(session_id, str(text), reply)
                    except Exception:
                        pass
                return reply
            task = str(route.text or "").strip() or str(text)
        else:
            task = str(text)

        reason = await self._ensure_worker()
        if reason:
            return self._worker_down_notice(task, reason)[1]

        self.agent = self._build_agent()
        res = await self.agent.ainvoke(
            {"messages": [*_lc_messages(hist), HumanMessage(content=task)]},
            config={"recursion_limit": self.config.recursion_limit},
        )
        msgs = res.get("messages", []) if isinstance(res, dict) else []
        reply = ""
        for m in reversed(list(msgs or [])):
            if isinstance(m, dict):
                role = m.get("type") or m.get("role") or ""
                content = m.get("content", "")
            else:
                role = getattr(m, "type", None) or ""
                content = getattr(m, "content", "")
            if role not in ("ai", "assistant"):
                continue
            if isinstance(content, list):
                content = " ".join(
                    str(c.get("text", c)) for c in content if isinstance(c, dict)
                )
            reply = str(content or "")
            break
        try:
            _thinking, answer = split_thinking(reply)
        except Exception:
            answer = reply
        reply = str(answer or reply or "").strip()
        if session_id and (text or reply):
            try:
                self.sessions.remember_turn(session_id, str(text), reply)
            except Exception:
                pass
        return reply

    # ---- voice agent (audio legs: fast path, full voice turns) ----

    @property
    def warmed(self) -> bool:
        """Whether warm() has completed at least once."""
        return self._warmed

    async def warm(self) -> "VoiceAgent":
        """Build every leg; collect failures instead of raising."""
        self.missing = []
        for name in ("vad", "stt", "tts", "front_llm", "llm"):
            if name == "tts" and not self.config.speak_text_turns:
                self.missing.append("tts leg: skipped (text-only)")
                continue
            if (
                name == "llm"
                and self.delegate_cfg is not None
                and not self._worker_eager
            ):
                continue
            leg = getattr(self.cascade, name, None) or getattr(self, name, None)
            if leg is None:
                continue
            try:
                await leg.warm()
                if name == "llm":
                    self._worker_warmed = True
            except Exception as exc:  # noqa: BLE001 - omit-and-report
                self.missing.append(f"{name} leg: {exc}")
        if (
            getattr(self, "llm", None) is None
            and getattr(self, "chat_model", None) is not None
        ):
            try:
                self._worker_warmed = True
            except Exception:
                pass
        try:
            await self.open_mcp_sessions()
        except Exception as exc:  # noqa: BLE001 - omit-and-report
            self.missing.append(f"mcp sessions: {exc}")
        self._warmed = True
        return self

    async def _voice_fast(self, audio, sr: int = 16000, session_id: str | None = None):
        """Fast voice turn via the cascade. Yields events.

        ..
        """
        async for ev in self.cascade.run_fast(audio, sr, session_id=session_id):
            yield ev

    async def _voice_turns(self, audio, sr: int = 16000, session_id: str | None = None):
        """Full voice turn via the cascade. Yields events.

        ..
        """
        async for ev in self.cascade.run_turns(audio, sr, session_id=session_id):
            yield ev
