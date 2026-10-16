# Console

**Operator event snapshot and tail**

The operator view behind `/talk` and `/chat`. Snapshot replays recent
events plus active calls and uptime; the stream replays history then
tails live. The terminal UI is `python -m src.server.console_cli`.

---
| Method | Path | Description |
|---|---|---|
| GET | `/console/snapshot` | Recent events plus active calls |
| WS | `/console/stream` | History replay then live tail |
---

## Objects

## ConsoleEvent
---
`ts` float
event timestamp in epoch seconds

`source` string
emitting side: talk, chat, turn

`kind` string
event kind: ring, user, agent, error

`text` string
event payload text

`sid` string
session the event belongs to
---

## Snapshot
---
`uptime_s` float
seconds since backend boot

`turns_done` integer
completed turns since boot

`active_calls` array
running calls as [{sid, state}]

`events` array
last 300 console events
---

## Each operation

GET    /console/snapshot
Returns uptime, turns done, active calls and the last 300 events.

REQUEST
```curl
curl http://127.0.0.1:8004/console/snapshot
```

RESPONSE
```json
{
  "uptime_s": 60.1,
  "turns_done": 3,
  "active_calls": [],
  "events": [
    {
      "ts": 123.4,
      "source": "chat",
      "kind": "agent",
      "text": "hi",
      "sid": "ses_abc123"
    }
  ]
}
```
----

WS    /console/stream
Replays history on connect, then pushes each new console event.

REQUEST
```json
{}
```

RESPONSE
```json
{
  "ts": 123.4,
  "source": "talk",
  "kind": "ring",
  "text": "",
  "sid": "ses_abc123"
}
```
----
