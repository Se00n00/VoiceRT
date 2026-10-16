# Voice

**One-shot STT, live STT stream, TTS**

Speech in and speech out around the agent. `/term/stt` VAD-gates then
transcribes one utterance; `/term/stt-stream` emits partials every 2 s
and one final; `/term/say` renders up to 500 chars via Kokoro.

---
| Method | Path | Description |
|---|---|---|
| POST | `/term/stt` | PCM utterance to transcript |
| WS | `/term/stt-stream` | Live PCM frames to partials plus final |
| POST | `/term/say` | Text to spoken wav |
---

## Objects

## SttResult
---
`kind` enum
result kind: text, empty, error

`text` string
transcribed utterance text

`message` string
human reason when kind is not text
---

## SayResult
---
`kind` enum
result kind: audio or error

`wav_b64` string
float32 wav bytes as base64

`sr` integer
sample rate of the rendered wav
---

## Each operation

POST    /term/say
Renders text to float32 wav b64 for playback. Fails loudly in text-only mode.

BODY
---
`text` string required
input text, 1-500 chars, truncated server-side
---

REQUEST
```curl
curl -X POST http://127.0.0.1:8004/term/say \
  -H 'Content-Type: application/json' \
  -d '{"text": "hello there"}'
```

RESPONSE
```json
{
  "kind": "audio",
  "wav_b64": "<float32 b64>",
  "sr": 24000
}
```
----

POST    /term/stt
Base64 PCM16 mono to transcript; silence reads as `empty`, never an error.

BODY
---
`pcm_b64` string required
raw int16 mono samples as base64

`sr` integer
sample rate of the samples, default 16000
---

REQUEST
```curl
curl -X POST http://127.0.0.1:8004/term/stt \
  -H 'Content-Type: application/json' \
  -d '{"pcm_b64": "<int16 b64>", "sr": 16000}'
```

RESPONSE
```json
{
  "kind": "text",
  "text": "hello there"
}
```
----

WS    /term/stt-stream
Binary int16 frames in; `partial` events out, `{"type": "stop"}` ends it.

REQUEST
```json
{
  "type": "hello",
  "sr": 16000
}
```

RESPONSE
```json
{
  "event": "final",
  "text": "hello there"
}
```
----
