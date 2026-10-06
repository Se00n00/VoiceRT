"""New-stack server tests: 3 endpoints + minimal /talk protocol (no weights)."""
import struct
import unittest


class _FakeVad:
    speech = False  # tests flip this to simulate live speech

    async def active(self, chunk, sr=16000):
        return _FakeVad.speech  # no auto-commit by default; commit explicit

    def reset(self):
        pass


class _FakeAgent:
    """Same surface as VoiceAgent.__call__ + vad, with canned streams."""

    # Seconds to stall at turn start (barge tests need a live turn to
    # interrupt; default 0 keeps every other test instant).
    turn_delay = 0.0

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
        import asyncio

        import numpy as np

        from src.agent.events import AgentEvent

        if _FakeAgent.turn_delay:
            await asyncio.sleep(_FakeAgent.turn_delay)
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
        self.assertEqual(paths, ["/contact/inbound", "/contact/reply",
                                 "/health", "/latency", "/metrics", "/talk",
                                 "/tg/connect", "/tg/creator-token",
                                 "/tg/pairing/new", "/tg/pairing/status",
                                 "/tg/qr", "/tg/status",
                                 "/tg/token", "/vad",
                                 "/wa/connect", "/wa/qr"])
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

    def test_barge_cancels_a_live_turn(self):
        c = _client()
        _FakeVad.speech = True
        _FakeAgent.turn_delay = 0.5
        try:
            with c.websocket_connect("/talk") as ws:
                ws.receive_json()  # ready
                ws.send_bytes(struct.pack("<160h", *([500] * 160)))
                ws.send_json({"type": "commit"})
                # speech while the slow turn runs -> barge, no done
                ws.send_bytes(struct.pack("<160h", *([500] * 160)))
                got_barge, got_done = False, False
                for _ in range(24):
                    msg = ws.receive_json()
                    if msg["event"] == "barge":
                        got_barge = True
                    if msg["event"] == "done":
                        got_done = True
                    if got_barge:
                        break
                self.assertTrue(got_barge)
                self.assertFalse(got_done)
                # socket still alive after the kill
                ws.send_json({"type": "reset"})
                msg = ws.receive_json()
                self.assertEqual(msg["event"], "node")
        finally:
            _FakeVad.speech = False
            _FakeAgent.turn_delay = 0.0

    def test_commit_while_busy_is_rejected(self):
        c = _client()
        _FakeAgent.turn_delay = 0.5
        try:
            with c.websocket_connect("/talk") as ws:
                ws.receive_json()  # ready
                ws.send_bytes(struct.pack("<160h", *([500] * 160)))
                ws.send_json({"type": "commit"})
                ws.send_bytes(struct.pack("<160h", *([500] * 160)))
                ws.send_json({"type": "commit"})
                errs = []
                for _ in range(24):
                    msg = ws.receive_json()
                    if msg["event"] == "error":
                        errs.append(msg["message"])
                    if msg["event"] == "done":
                        break
                self.assertIn("turn already running", errs)
        finally:
            _FakeAgent.turn_delay = 0.0

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

    def test_partial_opt_in_without_stt_leg_is_inert(self):
        # FakeAgent has no .stt: partial_s must be accepted and change
        # nothing (no partial frames, flow still reaches done).
        c = _client()
        with c.websocket_connect("/talk") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "config", "sr": 16000, "partial_s": 0.7})
            ready = ws.receive_json()
            self.assertEqual(ready["event"], "ready")
            ws.send_bytes(struct.pack("<160h", *([500] * 160)))
            ws.send_json({"type": "commit"})
            partials = 0
            for _ in range(16):
                msg = ws.receive_json()
                if (msg["event"] == "node" and msg.get("node") == "stt"
                        and msg.get("kind") == "partial"):
                    partials += 1
                if msg["event"] == "done":
                    break
            self.assertEqual(partials, 0)
            self.assertEqual(msg["event"], "done")

    def test_get_agent_env_backend_override(self):
        # Env knobs build the configured agent WITHOUT warming (no weights).
        import os

        import server as S

        S._agent = None
        os.environ["VOICE_LLM_BACKEND"] = "bonsai"
        os.environ["VOICE_FAST_VOICE"] = "1"
        try:
            agent = S.get_agent()
            self.assertEqual(agent.config.llm.backend, "bonsai")
            self.assertTrue(agent.config.fast_voice)
        finally:
            del os.environ["VOICE_LLM_BACKEND"]
            del os.environ["VOICE_FAST_VOICE"]
            S._agent = None


class _FakeSidecar:
    def __init__(self, prompt_ms=12.5, boom=False):
        self.calls = []
        self.prompt_ms = prompt_ms
        self.boom = boom

    def _payload(self, messages, tools, max_tokens, stop, stream):
        return {"messages": messages, "max_tokens": max_tokens,
                "stream": stream, "cache_prompt": True}

    def _post(self, path, payload, timeout):
        self.calls.append((path, payload))
        if self.boom:
            raise RuntimeError("slot busy")
        return {"timings": {"prompt_ms": self.prompt_ms,
                            "predicted_n": 0, "predicted_ms": 0.0}}


class _FakePrefillLLM:
    def __init__(self, backend=None):
        self._be = backend
        self.encoded = []

    def _backend(self):
        return self._be

    async def encode(self, messages, tools=None):
        self.encoded.append(messages)
        return [1, 2, 3]

    def messages_for_terminal(self, text, history=None, cwd="",
                              observation=""):
        msgs = [{"role": "system", "content": "sys"}]
        msgs.extend(history or [])
        msgs.append({"role": "user", "content": text})
        return msgs


class _FakeSessions:
    def history(self, sid):
        return [{"role": "user", "content": "prior"}]


class _FakePrefillAgent:
    def __init__(self, llm):
        self.llm = llm
        self.sessions = _FakeSessions()


class TestPrefill(unittest.TestCase):
    def test_messages_share_prefix_as_text_grows(self):
        import server as S

        ag = _FakePrefillAgent(_FakePrefillLLM())
        short = S.prefill_messages(ag, "s", "hello")
        long = S.prefill_messages(ag, "s", "hello world")
        self.assertIsNotNone(short)
        self.assertIsNotNone(long)
        # stable prefix (system + history), growing tail only
        self.assertEqual(short[:-1], long[:-1])
        self.assertTrue(long[-1]["content"].startswith(
            short[-1]["content"]))

    def test_messages_none_when_unusable(self):
        import server as S

        self.assertIsNone(S.prefill_messages(
            _FakePrefillAgent(None), "s", "hi"))
        ag = _FakePrefillAgent(_FakePrefillLLM())
        self.assertIsNone(S.prefill_messages(ag, "s", "  "))
        # fresh sid still primes (system prefix warms, history empty)
        fresh = S.prefill_messages(ag, None, "hi")
        self.assertIsNotNone(fresh)
        self.assertEqual(fresh[-1]["content"], "hi")

    def test_sidecar_prime_skipped(self):
        # Single-slot CPU sidecars prime slower than the pause window
        # (measured 2.4s) and would hold the only slot: no HTTP prime.
        import asyncio

        import server as S

        side = _FakeSidecar(prompt_ms=12.5)
        llm = _FakePrefillLLM(backend=side)
        self.assertIsNone(asyncio.run(
            S.prime_prefill(llm, [{"role": "user", "content": "hi"}])))
        self.assertEqual(side.calls, [])

    def test_encode_fallback_without_sidecar(self):
        import asyncio

        import server as S

        llm = _FakePrefillLLM(backend=None)
        ms = asyncio.run(S.prime_prefill(llm, [{"role": "user",
                                                "content": "hi"}]))
        self.assertEqual(ms, 0.0)
        self.assertEqual(len(llm.encoded), 1)

    def test_prime_failure_is_none(self):
        import asyncio

        import server as S

        llm = _FakePrefillLLM(backend=_FakeSidecar(boom=True))
        self.assertIsNone(asyncio.run(
            S.prime_prefill(llm, [{"role": "user", "content": "hi"}])))


if __name__ == "__main__":
    unittest.main()
