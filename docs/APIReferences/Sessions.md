# Sessions

**Persisted session title and transcript**

Model-free reads for resumes. `/term/title` serves the persisted name
(the TUI calls it on mount); `/term/history` replays user/assistant
turns straight off disk and touches the file against the sweeper.

---
| Method | Path | Description |
|---|---|---|
| GET | `/term/title` | Persisted session name |
| GET | `/term/history` | Full persisted transcript |
---

## Objects

## SessionHistory
---
`session_id` string
session the transcript belongs to

`title` string
persisted two-word session name

`messages` array
user/assistant turns as [{role, content}]
---

## Each operation

GET    /term/title
Returns the stored title or empty strings when sid is blank.

REQUEST
```curl
curl 'http://127.0.0.1:8004/term/title?sid=ses_abc123'
```

RESPONSE
```json
{
  "title": "Wifi setup",
  "session_id": "ses_abc123"
}
```
----

GET    /term/history
Returns title plus user/assistant messages for one session.

REQUEST
```curl
curl 'http://127.0.0.1:8004/term/history?sid=ses_abc123'
```

RESPONSE
```json
{
  "session_id": "ses_abc123",
  "title": "Wifi setup",
  "messages": [
    {
      "role": "user",
      "content": "hi"
    },
    {
      "role": "assistant",
      "content": "hello"
    }
  ]
}
```
----
