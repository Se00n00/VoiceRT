# Talk

**Duplex voice call over WebRTC**

One active call per process (4 GB box). Offer/answer over HTTP, mic Opus
in and reply Opus out over one transceiver, transcripts on the `voice`
datachannel. Speech while responding barges in and re-listens.

---
| Method | Path | Description |
|---|---|---|
| POST | `/talk/offer` | WebRTC handshake starting a call |
---

## Objects

## Offer
---
`sdp` string required
client WebRTC offer

`sid` string
session id, defaults to call-\<epoch\>
---

## Answer
---
`sdp` string
server WebRTC answer

`sid` string
session id for the call
---

## Each operation

POST    /talk/offer
Answers the offer and opens the call. 409 when voice is off or a call runs.

BODY
---
`sdp` string required
client WebRTC offer

`sid` string
session id, defaults to call-\<epoch\>
---

REQUEST
```curl
curl -X POST http://127.0.0.1:8004/talk/offer \
  -H 'Content-Type: application/json' \
  -d '{"sdp": "<webrtc offer>", "sid": "ses_abc123"}'
```

RESPONSE
```json
{
  "sdp": "<webrtc answer>",
  "sid": "ses_abc123"
}
```
----
