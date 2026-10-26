# 04 — Sessions

Identity, the conversation window, and what survives a restart.

## 4.1 Identity

The desktop client mints ids as `ses_` + 10 hex characters
(`newSessionId`, `src/tauri/src/app.tsx:15`). The Python side accepts
anything matching `_safe_sid` (`src/agent/memory.py:258`):
`[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. Anything else is hashed to
`sid-<sha256>` so a sid is always filesystem-safe.

The client generates the id and passes it on every turn; the server never
mints one for a WS connection. Defender-in-depth matters because the sid
is also a filename.

## 4.2 Who owns the window

`LongRunningAgent` holds conversation windows in memory only; persistence
is application work. The pair it exposes is deliberately narrow:

| Call | When | Signature |
|---|---|---|
| `import_session(sid, data)` | app start | `LongRunningAgent.import_session` |
| `export_session(sid)` | after each turn | `LongRunningAgent.export_session` |

`export_session` is just `history(sid)`. `import_session` appends
`{"role","content"}` dicts to the session store; `SessionMiddleware` converts
them to LangChain messages when it builds a model request. **Both never raise**:
a corrupt restore file degrades to an empty session instead of a dead
process.

## 4.3 The window store

The harness uses `JsonSessionMemory` (`src/agent/memory.py`), a
subclass of `LangChainSessionMemory` that writes one JSON file per
session atomically and reloads a cold sid from disk on next access. The
base class owns window/TTL/LRU semantics:

```
LangChainSessionMemory   RAM only, 20 turns / 30 min / 1000 sessions
JsonSessionMemory        one JSON file per session, atomic writes, TTL sweep on load
```

`LongRunningConfig` passes `max_session_turns=8`, `session_ttl_s=1800`,
`max_sessions=1000` and an optional `memory_tokens` ceiling. Eviction is
TTL first, then LRU at `max_sessions`, pruning the window to
`2 × max_turns`.

Sessions never hold GPU resources — only text — which is why stored
sessions may vastly outnumber in-flight turns. The scheduler and the
session store are orthogonal.

## 4.4 Serialization per session

`TurnScheduler.run` (`src/agent/long_running/scheduler.py`) takes a lock
keyed by `session_id`. Contenders are appended to `_pending` and drained
when the lock releases; the client sees a `term/queued` event. A second
turn queues and is injected as a follow-up instead of interleaving into a
window neither turn has seen fully.

`SessionMiddleware` (`src/agent/long_running/middleware/session.py`) is the
only reader/writer during a turn: it injects stored history into each model
request and calls `JsonSessionMemory.remember_turn` once in `after_agent`.

## 4.5 Client persistence

The Tauri client (`src/tauri/src/app.tsx`) stores:

- `localStorage["voicert.recents"]` — up to 20 ids + names, plus a pinned
  offline sample session.
- `localStorage["voicert.model"]` and `localStorage["voicert.theme"]` —
  restored on first contact.
- `?sid=<id>` in the URL — restored at boot.

Switching a recent calls `GET /term/history` to reload the whole
transcript, then `GET /term/title` when the store has no name. If the
backend has no record, the client falls back to the offline sample (for
the sample id) or `Untitled session`.

## 4.6 Titles

Two-word labels come from the small leg (`src/agent/title.py`): 12 tokens,
no tools, user text capped at 600 chars. `_maybe_title`
(`src/server/state.py:284`) fires after the first `/term` turn as a
background task so the title generation never delays the reply the user
is already reading. It stores via `set_title`/`get_title`
(`src/agent/memory.py:225-234`).

A dead leg or unparseable answer is not fatal: the title is decoration,
and the client falls back to the offline name.

## 4.7 Resuming

Open a recent from the sidebar or boot the web UI with `?sid=<id>`. The
client fetches the window from `/term/history`; the canonical JSON lives
under the sessions directory (`~/.local/share/voicert/sessions/` by
default). If the backend has no record, the app is responsible for
loading its own persisted JSON via `import_session`.

## 4.8 Known gaps

- `_maybe_title` is wired only to `/term`; the `/deep` path does not mint
  titles.

## 4.9 See also

- [03-memory.md](03-memory.md) — the tiers above the window
- [08-tui.md](08-tui.md) — the client that mints ids
