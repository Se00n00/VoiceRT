"""New-stack server tests: 3 endpoints + minimal /talk protocol (no weights)."""
import struct
import unittest


class _FakeVad:
    async def active(self, chunk, sr=16000):
        return False  # no auto-commit in tests; commit is explicit

    def reset(self):
        pass


class _FakeAgent:
    """Same surface as VoiceAgent.__call__ + vad, with canned streams."""

    def __init__(self):
        import numpy as np

        from engine import SessionStore

        self.vad = _FakeVad()
        self.sessions = SessionStore()
        self.warmed = True
        self.missing = []
        self._wav = np.zeros(240, dtype=np.float32)

    async def warm(self):
        return self

    async def __call__(self, audio, sr=16000, session_id=None):
        import numpy as np

        from src.agent.events import AgentEvent

        yield AgentEvent(node="vad", kind="segments",
                         data={"segments": [[0.0, 0.5]], "audio_dur_s": 1.0,
                               "speech_s": 0.5})
        yield AgentEvent(node="stt", kind="text",
                         data={"text": "hello", "rtf": 0.01, "ttfs": 0.01,
                               "dur_s": 1.0})
        yield AgentEvent(node="llm", kind="token",
                         data={"token": "Hi.", "first": True})
        yield AgentEvent(node="llm", kind="done", data={"text": "Hi."})
        yield AgentEvent(node="tts", kind="audio",
                         data={"wav": np.asarray(self._wav), "sr": 24000,
                               "sentence": "Hi.", "synth_s": 0.01})
        yield AgentEvent(node="turn", kind="summary",
                         data={"text": "hello", "reply": "Hi.",
                               "node_s": {}, "ttfa_s": 0.1, "total_s": 0.2,
                               "session_id": session_id})


def _client():
    from fastapi.testclient import TestClient

    import server as S
    from server import create_app

    S._agent = None
    return TestClient(create_app(agent=_FakeAgent()),
                      raise_server_exceptions=False)


class TestEndpoints(unittest.TestCase):
    def test_route_table_is_three(self):
        import server as S

        paths = sorted({r.path for r in S.router.routes})
        self.assertEqual(paths, ["/health", "/metrics", "/talk"])
        # browser harness lives on its own router (single-model, no sidecar)
        bpaths = sorted({r.path for r in S.browser_router.routes})
        self.assertEqual(bpaths, ["/browser/act", "/browser/tools", "/tts/say"])

    def test_openapi_lists_talk(self):
        # FastAPI skips WS routes in OpenAPI; server.py declares /talk
        # by hand so it stays visible in /docs and /openapi.json.
        paths = set(_client().get("/openapi.json").json()["paths"])
        self.assertTrue({"/health", "/metrics", "/talk"} <= paths)
        self.assertTrue({"/browser/act", "/browser/tools", "/tts/say"} <= paths)

    def test_health(self):
        r = _client().get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["nodes"], ["vad", "stt", "llm", "tts"])
        self.assertIn("uptime_s", body)

    def test_metrics_shape(self):
        r = _client().get("/metrics")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("turns", body)
        self.assertIn("events", body)
        self.assertIn("mean_s", body["turns"])

    def test_no_v1_routes(self):
        c = _client()
        self.assertEqual(c.get("/v1/sessions").status_code, 404)
        self.assertEqual(
            c.post("/v1/chat", json={"prompt": "hi"}).status_code, 404)


class TestTalkProtocol(unittest.TestCase):
    def test_ready_commit_done(self):
        c = _client()
        with c.websocket_connect("/talk") as ws:
            ready = ws.receive_json()
            self.assertEqual(ready["event"], "ready")
            self.assertIn("session_id", ready)
            # empty commit -> error, socket stays alive
            ws.send_json({"type": "commit"})
            err = ws.receive_json()
            self.assertEqual(err["event"], "error")
            # one audio chunk (no auto-commit: vad is False) then commit
            ws.send_bytes(struct.pack("<1600h", *([1000] * 1600)))
            ws.send_json({"type": "commit"})
            kinds = []
            for _ in range(16):
                msg = ws.receive_json()
                kinds.append((msg["event"], msg.get("node")))
                if msg["event"] == "done":
                    break
            self.assertIn(("node", "vad"), kinds)
            self.assertIn(("node", "stt"), kinds)
            self.assertIn(("node", "llm"), kinds)
            self.assertIn(("node", "tts"), kinds)
            done = msg
            self.assertEqual(done["summary"]["reply"], "Hi.")

    def test_tts_audio_is_b64(self):
        import base64

        c = _client()
        with c.websocket_connect("/talk") as ws:
            ws.receive_json()  # ready
            ws.send_bytes(struct.pack("<160h", *([500] * 160)))
            ws.send_json({"type": "commit"})
            wav_b64 = None
            for _ in range(16):
                msg = ws.receive_json()
                if msg["event"] == "node" and msg.get("node") == "tts":
                    wav_b64 = msg["data"].get("wav_b64")
                if msg["event"] == "done":
                    break
            self.assertIsNotNone(wav_b64)
            self.assertEqual(len(base64.b64decode(wav_b64)), 240 * 4)

    def test_reset_and_config(self):
        c = _client()
        with c.websocket_connect("/talk") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "config", "sr": 16000,
                          "session_id": "abc"})
            ready = ws.receive_json()
            self.assertEqual(ready["session_id"], "abc")
            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
            self.assertEqual(msg["event"], "node")
            self.assertEqual(msg["node"], "vad")

    def test_metrics_count_turns(self):
        c = _client()
        with c.websocket_connect("/talk") as ws:
            ws.receive_json()
            ws.send_bytes(struct.pack("<160h", *([500] * 160)))
            ws.send_json({"type": "commit"})
            for _ in range(16):
                if ws.receive_json()["event"] == "done":
                    break
        body = c.get("/metrics").json()
        self.assertGreaterEqual(body["turns"]["done"], 1)
        self.assertGreaterEqual(body["events"].get("tts", 0), 1)


if __name__ == "__main__":
    unittest.main()
