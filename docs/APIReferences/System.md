# System

**Liveness, model, brain and metrics**

Liveness plus model-leg switching and resource metrics. Nothing here
runs a turn; `GET /metrics` and `GET /legs` build the agent (and its
XDG bootstrap) on first call via `get_agent()`.

---
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/model` | Current model plus available profiles |
| POST | `/model/switch` | Hot-swap the LLM leg |
| GET | `/brain` | Voice vs harness brain mode |
| POST | `/brain/switch` | Switch brains |
| GET | `/metrics` | Uptime, turns, VRAM/RAM/CPU/GPU |
| GET | `/metrics/context` | Session context usage |
| GET | `/legs` | Leg labels, model, tool policy |
---

## Objects

## Health
---
`ok` boolean
always true when the bridge answers

`agent_loaded` boolean
whether the agent warmed without missing legs

`missing` string[]
leg names that failed to warm

`uptime_s` float
seconds since backend boot
---

## ModelProfile
---
`name` string
profile id used by `/model/switch`

`label` string
human label shown in the TUI

`backend` string
leg id: qwen, bonsai, or a cloud backend

`model` string
model id the profile resolves to

`desc` string
one-line profile description

`ready` boolean
whether the profile can be switched to

`current` boolean
whether this profile is active now
---

## Metrics
---
`uptime_s` float
seconds since backend boot

`turns.done` integer
completed turns since boot

`turns.total_s` float
sum of completed turn durations

`turns.last_s` float
duration of the last completed turn

`vram.allocated_mb` integer
CUDA bytes currently allocated

`vram.reserved_mb` integer
CUDA bytes reserved by the allocator

`vram.peak_mb` integer
peak CUDA allocation since boot

`vram.total_mb` integer
device VRAM total

`gpu_util` integer
nvidia-smi utilization percent
---

## Each operation

GET    /health
Returns ok plus warm state, never raises.

REQUEST
```curl
curl http://127.0.0.1:8004/health
```

RESPONSE
```json
{
  "ok": true,
  "uptime_s": 12.4,
  "agent_loaded": false,
  "missing": []
}
```
----

GET    /model
Lists every profile in `MODEL_PROFILES` with the current flag.

REQUEST
```curl
curl http://127.0.0.1:8004/model
```

RESPONSE
```json
{
  "current": "qwen",
  "available": [
    {
      "name": "qwen",
      "label": "qwen3-0.6B",
      "backend": "qwen",
      "model": "Qwen/Qwen3-0.6B",
      "desc": "0.6B fused on CUDA",
      "ready": true,
      "current": true
    }
  ]
}
```
----

POST    /model/switch
Hot-swaps the LLM leg, keeps sessions/TTS/STT/VAD, never raises.

BODY
---
`name` string required
profile id: qwen, bonsai, groq, anthropic, openai, openrouter, deepseek, openai_compat
---

REQUEST
```curl
curl -X POST http://127.0.0.1:8004/model/switch \
  -H 'Content-Type: application/json' \
  -d '{"name": "groq"}'
```

RESPONSE
```json
{
  "kind": "ok",
  "model": "groq"
}
```
----

GET    /brain
Returns the active brain plus the two available modes.

REQUEST
```curl
curl http://127.0.0.1:8004/brain
```

RESPONSE
```json
{
  "mode": "voice",
  "available": [
    {
      "name": "voice",
      "desc": "full voice agent on the main LLM harness",
      "current": true
    },
    {
      "name": "harness",
      "desc": "LLM harness only (text turns, no voice mode)",
      "current": false
    }
  ]
}
```
----

POST    /brain/switch
Switches brains; harness mode rejects `/talk` and `/term/say`.

BODY
---
`mode` string required
target brain: voice or harness
---

REQUEST
```curl
curl -X POST http://127.0.0.1:8004/brain/switch \
  -H 'Content-Type: application/json' \
  -d '{"mode": "harness"}'
```

RESPONSE
```json
{
  "kind": "ok",
  "mode": "harness"
}
```
----

GET    /metrics
Uptime, turn counters, VRAM/RAM/CPU plus GPU utilization.

REQUEST
```curl
curl http://127.0.0.1:8004/metrics
```

RESPONSE
```json
{
  "uptime_s": 60.1,
  "turns": {
    "done": 3,
    "total_s": 9.4,
    "last_s": 2.1
  },
  "queue": {
    "pending": 0,
    "locks": 0
  },
  "vram": {
    "allocated_mb": 0,
    "reserved_mb": 0,
    "peak_mb": 0,
    "total_mb": 4096,
    "cuda": false
  },
  "gpu_util": 0
}
```
----

GET    /metrics/context
Token estimate for one session against the active context window.

REQUEST
```curl
curl 'http://127.0.0.1:8004/metrics/context?sid=ses_abc123'
```

RESPONSE
```json
{
  "tokens": 128,
  "ctx": 4096,
  "pct": 0.031
}
```
----

GET    /legs
Leg labels, active model and the allowed tool-op policy.

REQUEST
```curl
curl http://127.0.0.1:8004/legs
```

RESPONSE
```json
{
  "legs": {
    "vad": "silero",
    "stt": "whisper",
    "llm": "qwen3-0.6B",
    "tts": "kokoro"
  },
  "model": {
    "name": "qwen",
    "label": "qwen3-0.6B",
    "backend": "qwen"
  },
  "policy": {
    "ops": ["exec", "read"],
    "deny": [],
    "confirm": ["exec", "read"]
  }
}
```
----
