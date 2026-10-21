# 08 — TUI

The desktop client: a Tauri shell around a Vite + SolidJS web app with a
React voice-mode island. It is a **pure client** — it never spawns the
backend.

## 8.1 Layout

```
src/tauri/
├── index.html                 # Vite entry
├── package.json               # scripts + deps
├── src/
│   ├── index.tsx              # mount
│   ├── app.tsx                # layout, commands, sessions, turn socket
│   ├── styles.css             # Tailwind + wallpaper
│   ├── api/client.ts          # HTTP + WS client for :8004
│   ├── audio/                 # recorder worklet, WebAudio, DSP
│   └── components/            # cards, voice overlay, metrics, keys, samples
└── src-tauri/                 # native shell (Rust + WebKitGTK)
```

The screen is the floating-cards layout: icon rail on the left, transcript
and composer in the centre, Get Started / model / Copy&Run cards on the
right.

Transport defaults to `http://127.0.0.1:8004`. Override it with
`VITE_VOICE_BRIDGE`, or persist a choice in
`localStorage["voicert.bridge"]`. Text turns use the persistent `WS /term`
socket (`TurnSocket`, `api/client.ts`), which also pumps the confirm gate.
Voice recording uses `WS /term/stt-stream`; playback uses `POST /term/say`.
History and titles come from `GET /term/history` and `GET /term/title`.

## 8.2 Commands

`runCommand` (`app.tsx:697`) is the single dispatcher:

```
/new       start a fresh session
/cwd       /cwd PATH — show or set the working directory
/clear     clear the transcript
/model     /model [name] — list or switch the model
/brain     /brain [voice|harness] — switch the active brain
/voice     /voice [sec] — record a full voice turn (voice brain only)
/dictate   /dictate [sec] — transcribe into the composer
/mic       /mic [sec] — short dictation
/sample    /sample [name] — insert a bundled conversation
/help      show the command list
```

Tab completes commands; Enter sends exactly what is typed. Unknown
commands get an edit-distance suggestion, so `/modle` points at `/model`.

## 8.3 Voice

`/brain voice` switches to the full duplex voice agent; `/brain harness`
keeps the LLM harness and disables the voice overlay, while dictation
still works. The overlay is `ReactVoiceMode` (React) mounted inside the
Solid app; mic capture runs in an AudioWorklet
(`audio/recorder-worklet.js`), and playback streams TTS chunks through
WebAudio.

`/dictate` and `/mic` record for a bounded number of seconds, send PCM16
to `/term/stt`, and put the transcript in the composer instead of sending
a turn. `/voice` sends the recording as a turn.

## 8.4 Sessions

The client mints ids as `ses_` + 10 hex characters
(`newSessionId`, `app.tsx:15`). Recents live in
`localStorage["voicert.recents"]` (max 20), with a pinned offline sample
session. Switching a recent calls `/term/history` to restore the whole
transcript, then `/term/title` if the store has no title.

The backend still owns the canonical session files; the client only
caches the sidebar list and the model/theme choices.

## 8.5 Shortcuts

| Keys | Action |
|---|---|
| `Ctrl+V` | toggle voice mode (outside text fields) |
| `Ctrl+S` | toggle the sessions/conversation panel |
| `Ctrl+Shift+N` | new session |
| `T` | start dictation |
| `E` | toggle metrics |
| `?` | shortcuts modal |

Bare keys are ignored while typing, so paste and normal editing are never
hijacked.

## 8.6 Theme and samples

`app.tsx` stores `light`/`dark` in `localStorage["voicert.theme"]`; the
model choice is restored on first contact so a restart does not re-pick a
backend. `/sample` inserts one of the bundled conversations (including a
tool/confirm chain) for demos without a backend.

## 8.7 Build and checks

```bash
cd src/tauri
npm install
npm run typecheck
npm run build
npm run dev            # web UI at Vite's port
npm run tauri:dev      # native shell (needs Rust + WebKitGTK)
```

The desktop shell is optional; the web UI is enough against a running
`python -m src.server` on `:8004`.

## 8.8 See also

- [06-processes.md](06-processes.md) — the `:8004` backend
- [04-sessions.md](04-sessions.md) — ids and resume
