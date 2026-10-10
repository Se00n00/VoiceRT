"""Terminal bridge: FastAPI service for the Ink TUI (stock React/Ink).

Runs the unified VoiceAgent (voice + terminal turns over one deepagents
loop) and exposes it over HTTP + WebSocket on :8004. Run ONE backend
process (same 4GB GPU).

Endpoints (see ``src/server/routes/`` for the handlers):
- ``GET /health`` — {ok, agent_loaded, missing}
- ``GET /model`` — {current, available:[{name, model, backend, desc, ready}]}
- ``POST /model/switch`` {name} — hot-swap the LLM leg (VRAM-safe)
- ``POST /term/stt`` {pcm_b64, sr} — raw mic PCM -> {text}
- ``WS /term/stt-stream`` — live mic PCM16 mono frames in, partial/final
  transcripts out
- ``POST /term/say`` {text} — text -> {wav_b64, sr} (Kokoro)
- ``GET /term/history?sid=`` — past session messages
- ``GET /term/title?sid=`` — persisted session name (no model call)
- ``WS /term`` — full turn + confirm gate
- ``WS /deep`` — autonomous turns over the same agent (confirms auto-pass)
- ``POST /talk/offer`` — WebRTC offer/answer handshake for a duplex call
- ``WS /chat`` — pure Qwen chat (no tools, no confirms)
- ``GET /metrics`` — uptime, turns, VRAM/RAM/CPU/GPU, legs
- ``GET /metrics/context?sid=`` — session context usage
- ``GET /legs`` — leg labels, model profile, tool policy
- ``GET /console/snapshot`` — recent console events + active calls
- ``WS /console/stream`` — live console event tail
  (terminal UI: ``python -m src.server.console_cli``)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import state
from .routes import (
    chat_router,
    console_router,
    sessions_router,
    system_router,
    talk_router,
    turns_router,
    voice_router,
)

__all__ = ["create_app", "app"]


def create_app(agent=None):
    if agent is not None:
        state._agent = agent
    app = FastAPI(title="voice-term bridge")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system_router)
    app.include_router(sessions_router)
    app.include_router(voice_router)
    app.include_router(turns_router)
    app.include_router(chat_router)
    app.include_router(talk_router)
    app.include_router(console_router)

    @app.on_event("startup")
    async def _warm():
        ag = state.get_agent()
        try:
            await ag.warm()
            print("bridge ready", flush=True)
            if getattr(ag, "missing", []):
                print(f"bridge missing: {ag.missing}", flush=True)
        except Exception as exc:
            print(f"bridge not warmed: {exc}", flush=True)

    return app


app = create_app()
