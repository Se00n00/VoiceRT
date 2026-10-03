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

## 6.2 `bridge.py` — the TUI backend

Endpoints (`bridge.py`):

| Route | Line | Purpose |
|---|---|---|
| `GET /health` | 419 | liveness, warm state, missing legs |
| `GET /model` | 431 | current model profile |
| `POST /model/switch` | 441 | hot-swap the LLM leg |
| `GET /term/title` | 445 | session title |
| `POST /term/stt` | 462 | audio → text |
| `POST /term/say` | 488 | text → audio |
| `WS /term` | 502 | the terminal turn stream |
| `WS /deep` | 605 | the deep-agent turn stream |

Supporting pieces: `_build_llm` (150), `_free_llm_gpu` (158),
`_maybe_title` (187), `_compact_sessions` (232), `switch_model` (285),
`get_agent` (368), `_wav_b64` (401), `create_app` (406).

`/term` runs one turn at a time per session and keeps a pending queue with a
`Queue` plus per-turn `Event`s (`bridge.py:508-524`), timing out at 120 s.
Turn work is dispatched with `asyncio.create_task` (`bridge.py:558`, `597`,
`658`).

`/model/switch` to an explicitly warmed leg resets the lazy-worker latch
(`bridge.py:343-349`); if the new sidecar only attached to the old server,
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
async def _warm():        # bridge.py:646-647
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
ps aux | grep -E 'llm_chat|test_toolcall|server.py|bridge.py'
```

## 6.6 See also

- [01-agent.md](01-agent.md) — what `create_app` wraps
- [08-tui.md](08-tui.md) — the client
