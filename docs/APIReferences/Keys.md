# Keys

**Store, list and delete provider keys**

OS-keychain key management that never returns values. `GET` reports
per-provider stored flags; writes are loopback-only (127.0.0.1 / ::1)
even though the bridge CORS policy is permissive.

---
| Method | Path | Description |
|---|---|---|
| GET | `/keys` | Per-provider stored flags |
| POST | `/keys` | Store one provider key |
| DELETE | `/keys/{provider}` | Remove one provider key |
---

## Objects

## KeyStatus
---
`providers` object
per-backend stored flags, never values

`store` string
keychain store name

`available` boolean
whether the OS keychain is usable
---

## KeyBody
---
`provider` string required
backend id: groq, gemini, anthropic, openai, openrouter, deepseek, openai_compat

`key` string required
API key, never echoed back
---

## Each operation

GET    /keys
Reports which providers have keys plus the keychain store name.

REQUEST
```curl
curl http://127.0.0.1:8004/keys
```

RESPONSE
```json
{
  "providers": {
    "groq": true,
    "gemini": false,
    "anthropic": false,
    "openai": false,
    "openrouter": false,
    "deepseek": false,
    "openai_compat": false
  },
  "store": "voicert",
  "available": true
}
```
----

POST    /keys
Stores one key in the OS keychain; never echoes the value back.

BODY
---
`provider` string required
backend id: groq, gemini, anthropic, openai, openrouter, deepseek, openai_compat

`key` string required
API key, never echoed back
---

REQUEST
```curl
curl -X POST http://127.0.0.1:8004/keys \
  -H 'Content-Type: application/json' \
  -d '{"provider": "groq", "key": "gsk-..."}'
```

RESPONSE
```json
{
  "kind": "ok",
  "provider": "groq",
  "stored": true
}
```
----

DELETE    /keys/{provider}
Removes one provider key from the OS keychain. Loopback-only.

REQUEST
```curl
curl -X DELETE http://127.0.0.1:8004/keys/groq
```

RESPONSE
```json
{
  "kind": "ok",
  "provider": "groq",
  "cleared": true
}
```
----
