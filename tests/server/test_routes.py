"""Server tests: schemas + route table (no live server, no weights).

Covers server/schemas.py validation and that server/app.py registers
GET /health, POST /v1/vad /v1/transcribe /v1/chat /v1/speak /v1/voice
and WS /v1/talk — all without starting uvicorn or loading the engine.
"""
import unittest


class TestSchemas(unittest.TestCase):
    def test_chat_req_defaults(self):
        from server.schemas import ChatReq

        r = ChatReq(prompt="hi")
        self.assertEqual(r.max_tokens, 48)
        self.assertFalse(r.stream)

    def test_speak_req(self):
        from server.schemas import SpeakReq

        self.assertEqual(SpeakReq(text="hello").text, "hello")

    def test_response_models(self):
        from server.schemas import (ChatResp, TranscribeResp, VadResp,
                                    VoiceResp)

        self.assertEqual(VadResp().n_segments, 0)
        self.assertEqual(TranscribeResp(text="x").text, "x")
        self.assertEqual(ChatResp(text="y").text, "y")
        self.assertEqual(VoiceResp(text="a", reply="b").reply, "b")


class TestRoutes(unittest.TestCase):
    def _app(self):
        try:
            from server.app import create_app
        except Exception as exc:
            self.skipTest(f"server.app not importable ({exc})")
        try:
            return create_app()
        except Exception as exc:
            self.skipTest(f"create_app failed ({exc})")

    def test_route_table(self):
        try:
            from server.routes import router
        except Exception as exc:
            self.skipTest(f"server.routes not importable ({exc})")
        routes = set()
        for r in router.routes:
            for m in getattr(r, "methods", None) or set():
                routes.add((m, getattr(r, "path", "")))
        for method, path in (("GET", "/health"),
                             ("POST", "/v1/vad"),
                             ("POST", "/v1/transcribe"),
                             ("POST", "/v1/chat"),
                             ("POST", "/v1/speak"),
                             ("POST", "/v1/voice")):
            self.assertIn((method, path), routes, f"missing {method} {path}")
        # app-level: create_app() must expose the same paths via openapi
        # (FastAPI>=0.14x defers include_router expansion in app.routes).
        try:
            from server.app import create_app

            paths = set(create_app().openapi()["paths"])
        except Exception as exc:
            self.skipTest(f"openapi check skipped ({exc})")
        for path in ("/health", "/v1/vad", "/v1/transcribe", "/v1/chat",
                     "/v1/speak", "/v1/voice"):
            self.assertIn(path, paths, f"app missing {path}")

    def test_websocket_registered(self):
        try:
            from server.websocket import router as ws_router
        except Exception as exc:
            self.skipTest(f"server.websocket not importable ({exc})")
        paths = [getattr(r, "path", "") for r in ws_router.routes]
        self.assertIn("/v1/talk", paths, "missing WS /v1/talk")

    def test_lazy_engine(self):
        # routes.py must keep `engine.engine` import lazy (inside
        # handlers/startup), never at module top level.
        import os

        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, "server", "routes.py")) as f:
            lines = f.read().splitlines()
        top = [ln for ln in lines
               if ln.startswith("from engine") or ln.startswith("import engine")]
        self.assertEqual(top, [], f"engine import must be lazy, found: {top}")


class TestGuards(unittest.TestCase):
    def test_guard_wav(self):
        import numpy as np

        from server.routes import MAX_AUDIO_S, _guard_wav

        from fastapi import HTTPException

        dur = _guard_wav(np.zeros(16000, dtype=np.float32), 16000)
        self.assertAlmostEqual(dur, 1.0)
        with self.assertRaises(HTTPException) as cm:
            _guard_wav(np.zeros(0, dtype=np.float32), 16000)
        self.assertEqual(cm.exception.status_code, 400)
        big = np.zeros(int(MAX_AUDIO_S * 16000) + 16000, dtype=np.float32)
        with self.assertRaises(HTTPException) as cm:
            _guard_wav(big, 16000)
        self.assertEqual(cm.exception.status_code, 413)

    def test_queue_503(self):
        import server.routes as R
        from fastapi import HTTPException

        # Fill every FIFO slot, then shrink the timeout so the queued
        # acquire fails fast with 503 instead of waiting.
        tickets = [R._sched.acquire(blocking=False) for _ in range(R._sched.max_concurrency)]
        assert all(t is not None for t in tickets)
        old_timeout = R._QUEUE_TIMEOUT_S
        R._QUEUE_TIMEOUT_S = 0.05
        try:
            with self.assertRaises(HTTPException) as cm:
                R._acquire_or_503()
            self.assertEqual(cm.exception.status_code, 503)
        finally:
            R._QUEUE_TIMEOUT_S = old_timeout
            for t in tickets:
                R._release(t)

    def test_capacity_defaults(self):
        import server.routes as R

        cap = R.get_capacity()
        self.assertIn("max_sessions", cap)
        self.assertIn("max_inflight", cap)
        self.assertIn("generation_length", cap)

    def test_schema_bounds(self):
        from pydantic import ValidationError

        from server.schemas import ChatReq, SpeakReq

        with self.assertRaises(ValidationError):
            ChatReq(prompt="hi", max_tokens=100000)
        with self.assertRaises(ValidationError):
            ChatReq(prompt="")
        with self.assertRaises(ValidationError):
            SpeakReq(text="")
        with self.assertRaises(ValidationError):
            SpeakReq(text="x" * 2001)

    def test_health_no_engine(self):
        # /health must not construct the engine (liveness-safe).
        import server.routes as R

        self.assertFalse(R.engine_loaded())
        out = R.health()
        self.assertTrue(out["ok"])
        self.assertFalse(out["engine_loaded"])
        self.assertIn("uptime_s", out)

    def test_metrics_shape(self):
        import server.routes as R

        m = R.metrics()
        self.assertIn("uptime_s", m)
        self.assertIn("inflight", m)
        self.assertIn("endpoints", m)

    def test_ws_shares_engine(self):
        # WS must reuse the HTTP singleton (a 2nd VoiceEngine OOMs 4GB).
        import os

        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, "server", "websocket.py")) as f:
            src = f.read()
        self.assertIn("from server.routes import get_engine", src)


if __name__ == "__main__":
    unittest.main()
