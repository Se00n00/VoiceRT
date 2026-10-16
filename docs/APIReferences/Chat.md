# Chat

**Pure Qwen chat without tools**

Chunked text chat on the fused leg only. No tools, no confirms, no
delegate, and it refuses unless the qwen profile is active (so weights
never double-load next to voice). Turns append to the shared session.

---

| Method | Path | Description |
|---|---|---|
| WS | `/chat` | Token-streamed Qwen reply |

---

## Objects

## ChatFrame

---

`type` enum required
frame kind: chat

`text` string required
user message text

`session_id` string
session the turn appends to

---

## ChatReply
---
`event` enum
event kind: token, reply, error

`piece` string
word-chunk of the streaming reply

`reply` string
full final reply text

`message` string
error reason when event is error
---

## Each operation

WS    /chat
Sends word-chunk `token` events then one `reply`; errors when qwen is idle.

REQUEST
```json
{
  "type": "chat",
  "text": "what is voicert",
  "session_id": "ses_abc123"
}
```

RESPONSE
```json
{
  "event": "reply",
  "reply": "a local-first voice agent"
}
```
----
