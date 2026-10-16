# Turns

**Gated terminal turns and autonomous turns**

The two turn sockets over one deep-agent loop. `/term` pauses on a
`confirm` event for the client gate (120 s timeout); `/deep`
auto-passes confirms. One running turn per socket; `cancel` kills it.

---
| Method | Path | Description |
|---|---|---|
| WS | `/term` | Gated terminal turn with confirm gate |
| WS | `/deep` | Autonomous turn, confirms auto-pass |
---

## Objects

## TurnFrame
---
`type` enum required
frame kind: turn, confirm, cancel

`text` string
turn input text

`session_id` string
session the turn runs in

`cwd` string
working directory for tool execution

`ok` boolean
confirm verdict sent back to the gate
---

## TurnEvent
---
`event` enum
event kind: action, observation, token, thinking, chat, summary, audio, confirm, title, error, cancelled

`action` object
tool call the agent wants to run

`observation` string
tool result text

`reply` string
final assistant reply text

`piece` string
streamed reply chunk

`wav_b64` string
spoken reply audio as base64
---

## Each operation

WS    /term
Streams term events; emits `confirm` and waits for `{type: confirm, ok}`.

REQUEST
```json
{
  "type": "turn",
  "text": "list the repo root",
  "session_id": "ses_abc123",
  "cwd": "/tmp/work"
}
```

RESPONSE
```json
{
  "event": "chat",
  "reply": "here is the listing"
}
```
----

WS    /deep
Same loop as `/term` but confirms auto-pass; used by the TUI agent mode.

REQUEST
```json
{
  "type": "turn",
  "text": "rename the log file",
  "session_id": "ses_abc123",
  "cwd": "/tmp/work"
}
```

RESPONSE
```json
{
  "event": "summary",
  "reply": "renamed voicert.log to voicert.log.1"
}
```
----
