# 06 — Processes

Three entry points, three ports, three jobs. They do not share state — each
holds its own `VoiceAgent` and therefore its own copy of the weights.

## 6.1 Which process is which

| Command | Port | Purpose |
|---|---|---|
| `PYTHONPATH=. .venv/bin/python bridge.py` | 8004 | TUI backend |
| `PYTHONPATH=. .venv/bin/python server.py` | 8003 | voice API |
| `.venv/bin/python -u llm_chat.py` | — | text REPL |
| `cd src/tui && npm run dev` | — | OpenTUI client |

**Run one at a time.** Two agents = two weight copies = CUDA OOM on a 4 GB
card.

## 6.2 `src/server` — the TUI backend

Endpoints (`src/server/routes/`):

| Route | File | Purpose |
|---|---|---|
| `GET /health` | `system.py` | liveness, warm state, missing legs |
| `GET /model` | `system.py` | current model profile |
| `POST /model/switch` | `system.py` | hot-swap the LLM leg |
| `GET /term/title` | `sessions.py` | session title |
| `GET /term/history` | `sessions.py` | past session transcript |
| `POST /term/stt` | `voice.py` | audio → text |
| `WS /term/stt-stream` | `voice.py` | live partial transcripts |
| `POST /term/say` | `voice.py` | text → audio |
| `WS /term` | `turns.py` | the terminal turn stream |
| `WS /deep` | `turns.py` | the deep-agent turn stream |

Supporting pieces (`src/server/state.py`): `_build_llm`, `_free_llm_gpu`,
`_maybe_title`, `_compact_sessions`, `switch_model`, `get_agent`,
`_wav_b64`; `create_app` lives in `src/server/__init__.py`.

`/term` runs one turn at a time per session and keeps a pending queue with a
`Queue` plus per-turn `Event`s (`turns.py`, timing out at 120 s).
Turn work is dispatched with `asyncio.create_task`.

`/model/switch` to an explicitly warmed leg resets the lazy-worker latch
(`state.py: switch_model`); if the new sidecar only attached to the old server,
it re-warms after the old close so the swap never commits a dead port.

`/health` must stay liveness-safe — it answers before the agent exists, which
is what lets the TUI show `bridge down` versus `warming` correctly.

## 6.3 `server.py` — the voice API

Routes on `router` (`server.py`):

| Route | Line |
|---|---|
| `GET /health` | 172 |
| `GET /vad` | 260 |
| `GET /metrics` | 271 |
| `POST /contact/reply` | 309 |
| `POST /contact/inbound` | 326 |
| `GET /wa/connect`, `/wa/qr` | 344, 355 |
| `GET /tg/connect`, `/tg/status` | 378, 389 |
| `POST /tg/creator-token`, `/tg/pairing/new`, `/tg/token` | 417, 435, 484 |
| `GET /tg/pairing/status`, `/tg/qr` | 470, 502 |
| `WS /talk` | 526 |

The contact routes are the Telegram/WhatsApp surface:
`src/agent/contacts.py` (666 lines) holds the legs — `telegram`, `baileys`, and
the legacy `callmebot` — and `src/agent/pairing.py` (395) plus
`src/agent/tg_poll.py` (79) handle creator-token pairing and the long-poll
loop.

`WS /talk` is the voice turn path: one socket is one session, PCM16 mono in,
node events out.

## 6.4 The startup-warm trap

This one bit us and is worth writing down.

```python
@app.on_event("startup")
async def _warm():        # src/server/__init__.py
    await agent.warm()
```

Uvicorn **awaits the lifespan startup hook before it creates the listening
socket**. So a warm that hangs means no listener at all — not a slow start, a
silent port.

Observed on 2026-09-23: four bridge processes, `nvidia-smi` showing no compute
process, six HTTPS sockets in `CLOSE_WAIT` against what resolves to a
HuggingFace CDN, and `:8004` refusing connections. The warm was stuck
downloading model weights from inside the lifespan hook.

The fix is to bind first and warm second — a background task plus an explicit
`warming` / `warmed` / `warm_error` state on `/health`, and a bounded warm so a
hung download cannot wedge startup. **Not implemented yet.** Until it is, a
hung warm looks exactly like a dead bridge to the TUI.

## 6.5 Process hygiene

Model subprocesses (`arecord`, `aplay`, docker clients) die with their parents.
When in doubt:

```bash
ps aux | grep -E 'llm_chat|test_toolcall|src.server'
```

## 6.6 See also

- [01-agent.md](01-agent.md) — what `create_app` wraps
- [08-tui.md](08-tui.md) — the client
