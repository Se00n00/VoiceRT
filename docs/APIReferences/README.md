# API References

One file per route group on the `:8004` backend. Base URL is
`http://127.0.0.1:8004` (WS endpoints use `ws://`).

| File | Routes |
|---|---|
| [System](System.md) | `/health /model /model/switch /brain /brain/switch /metrics?sid=` |
| [Turns](Turns.md) | `WS /term WS /deep` |
| [Voice](Voice.md) | `/term/stt WS /term/stt-stream /term/say` |
| [Talk](Talk.md) | `/talk/offer` |
| [Chat](Chat.md) | `WS /chat` |
| [Sessions](Sessions.md) | `/term/title /term/history` |
| [Keys](Keys.md) | `/keys` |
