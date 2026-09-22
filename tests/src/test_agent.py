"""New-stack tests: dataclass configs, facades, turn graph, VoiceAgent.

No weights, no GPU: model legs are faked at the clean-class boundary.
"""
import unittest


class TestConfigs(unittest.TestCase):
    def test_dataclass_defaults(self):
        from src.main import VoiceAgentConfig
        from src.models.llm import LlmConfig
        from src.models.stt import SttConfig
        from src.models.tts import TtsConfig
        from src.models.vad import VadConfig

        cfg = VoiceAgentConfig()
        self.assertIsInstance(cfg.vad, VadConfig)
        self.assertIsInstance(cfg.stt, SttConfig)
        self.assertIsInstance(cfg.llm, LlmConfig)
        self.assertIsInstance(cfg.tts, TtsConfig)
        self.assertEqual(cfg.vad.threshold, 0.5)
        self.assertEqual(cfg.llm.max_tokens, 48)
        self.assertEqual(cfg.max_audio_s, 60.0)
        # overrides bind in code, not YAML
        custom = VoiceAgentConfig(vad=VadConfig(threshold=0.7))
        self.assertEqual(custom.vad.threshold, 0.7)

    def test_no_yaml_in_new_code(self):
        import os

        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        paths = [os.path.join(root, "src", "main.py"),
                 os.path.join(root, "server.py")]
        for top in (os.path.join(root, "src", "models"),
                    os.path.join(root, "src", "agent"),
                    os.path.join(root, "engine")):
            paths += [os.path.join(d, f)
                      for d, _, fs in os.walk(top) for f in fs
                      if f.endswith(".py")]
        self.assertGreater(len(paths), 10)
        for path in paths:
            with open(path) as f:
                body = f.read()
            self.assertNotIn("import yaml", body, path)
            self.assertNotIn("configs/", body, path)


class TestFacades(unittest.TestCase):
    def test_runtime_facade(self):
        import src.models.runtime.scheduler as real

        import src.models.runtime as nested

        self.assertIs(nested.FIFOScheduler, real.FIFOScheduler)
        self.assertIs(nested.scheduler.FIFOScheduler, real.FIFOScheduler)
        self.assertIs(nested.Profiler, __import__(
            "src.models.runtime.profiler", fromlist=["Profiler"]).Profiler)

    def test_triton_location(self):
        try:
            import src.models.triton_kernels as tk
            import src.models.triton_kernels.qwen as real
        except Exception as exc:
            self.skipTest(f"triton stack not importable ({exc})")
        self.assertIs(tk.qwen, real)
        self.assertTrue(real.HAVE_TRITON_KERNELS is False
                        or callable(real.fused_qkv_gqa))

    def test_engine_location(self):
        from engine.session import SessionStore as RealStore

        from engine import SessionStore, new_session_id

        self.assertIs(SessionStore, RealStore)
        self.assertEqual(len(new_session_id()), 32)


class TestTurnGraph(unittest.TestCase):
    def test_graph_nodes(self):
        from src.main import VoiceAgent

        nodes = set(VoiceAgent()._compiled().get_graph().nodes)
        self.assertTrue({"vad", "stt", "respond", "silence"} <= nodes)

    def test_router(self):
        from src.agent.nodes import route_after_vad

        self.assertEqual(route_after_vad({"segments": [[0.0, 0.5]]}), "stt")
        self.assertEqual(route_after_vad({"segments": []}), "silence")
        self.assertEqual(route_after_vad({}), "silence")

    def test_vad_node_unit(self):
        import asyncio

        import numpy as np

        from src.agent.nodes import vad_node

        events = []
        update = asyncio.run(vad_node(
            {"audio": np.zeros(16000, dtype=np.float32), "sr": 16000,
             "node_s": {}},
            vad=_FakeVad([(0.0, 0.5)]), writer=events.append))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].node, events[0].kind),
                         ("vad", "segments"))
        self.assertEqual(update["segments"], [[0.0, 0.5]])
        self.assertIn("vad", update["node_s"])

    def test_silence_node_unit(self):
        import asyncio
        import time

        from src.agent.nodes import silence_node

        events = []
        update = asyncio.run(silence_node(
            {"sid": "s", "t0": time.perf_counter(), "node_s": {}},
            writer=events.append))
        kinds = [(e.node, e.kind) for e in events]
        self.assertEqual(kinds, [("stt", "text"), ("turn", "summary")])
        self.assertTrue(update["silent"])
        self.assertEqual(events[-1].data["reply"], "")


class _FakeVad:
    def __init__(self, segs):
        self._segs = segs
        self.config = type("C", (), {"threshold": 0.5})()

    async def segments(self, audio, sr=16000):
        from src.models.vad import VadSegments

        return VadSegments(segments=tuple(self._segs),
                           audio_dur_s=len(audio) / float(sr))

    async def active(self, chunk, sr=16000):
        return bool(self._segs)

    def reset(self):
        pass


class _FakeStt:
    async def transcribe(self, audio, sr=16000):
        from src.models.stt import SttResult

        return SttResult(text="hello", rtf=0.01, ttfs=0.01,
                         dur_s=len(audio) / float(sr))


class _FakeLlm:
    def messages(self, text, history=None):
        return [{"role": "user", "content": text}]

    async def decode(self, ids):
        return "Hi there."

    async def stream(self, messages, max_tokens=None):
        from src.models.llm import LlmToken

        yield LlmToken(token_id=1, piece="Hi ", first=True)
        yield LlmToken(token_id=2, piece="there.", first=False)


class _FakeTts:
    def __init__(self):
        import numpy as np

        self._wav = np.zeros(240, dtype=np.float32)

    async def speak(self, text):
        from src.models.tts import TtsAudio

        return TtsAudio(wav=self._wav, sample_rate=24000, sentence=text,
                        synth_s=0.01)


def _agent_with(segs):
    from src.main import VoiceAgent

    ag = VoiceAgent()
    ag.vad, ag.stt, ag.llm, ag.tts = (_FakeVad(segs), _FakeStt(),
                                      _FakeLlm(), _FakeTts())
    return ag


class TestThinking(unittest.TestCase):
    def test_split_closed(self):
        from src.models.llm import split_thinking

        th, ans = split_thinking("<think>plan A then B</think>Done.")
        self.assertEqual(th, "plan A then B")
        self.assertEqual(ans, "Done.")

    def test_split_unclosed(self):
        from src.models.llm import split_thinking

        th, ans = split_thinking("<think>hmm, let me see")
        self.assertEqual(th, "hmm, let me see")
        self.assertEqual(ans, "")

    def test_split_none(self):
        from src.models.llm import split_thinking

        th, ans = split_thinking("Hi there.")
        self.assertEqual((th, ans), ("", "Hi there."))

    def test_split_case_insensitive(self):
        from src.models.llm import split_thinking

        th, ans = split_thinking("<THINK>  spaced  </THINK>  ok")
        self.assertEqual((th, ans), ("spaced", "ok"))

    def test_voice_thinking_not_spoken(self):
        import asyncio
        import time

        import numpy as np

        from src.agent.nodes import respond_node
        from src.models.llm import LlmToken

        class ThinkLlm(_FakeLlm):
            async def decode(self, ids):
                return "<think>choose greeting</think>Hi there."

            async def stream(self, messages, max_tokens=None):
                for i, piece in enumerate(["<think>choose ", "greeting</think>",
                                           "Hi ", "there."]):
                    yield LlmToken(token_id=10 + i, piece=piece, first=(i == 0))

        tts = _FakeTts()
        spoken = []
        orig = tts.speak

        async def spy(text):
            spoken.append(text)
            return await orig(text)

        tts.speak = spy  # type: ignore
        events = []
        state = {"text": "hi", "history": [], "sid": "s",
                 "remember": False, "node_s": {},
                 "t0": time.perf_counter()}
        asyncio.run(respond_node(state, llm=ThinkLlm(), tts=tts,
                                 sessions=None, writer=events.append))
        kinds = [(e.node, e.kind) for e in events]
        self.assertIn(("llm", "thinking"), kinds)
        think = [e for e in events if e.kind == "thinking"][0]
        self.assertIn("choose greeting", think.data["text"])
        # thoughts never reach TTS; answer does, whole
        self.assertTrue(spoken)
        self.assertFalse(any("choose greeting" in s for s in spoken))
        self.assertIn(("llm", "done"), kinds)
        done = [e for e in events if e.kind == "done"][0]
        self.assertEqual(done.data["text"], "Hi there.")
        self.assertEqual(events[-1].data["reply"], "Hi there.")

    def test_voice_no_think_unchanged(self):
        import asyncio
        import time

        from src.agent.nodes import respond_node

        tts = _FakeTts()
        events = []
        state = {"text": "hi", "history": [], "sid": "s",
                 "remember": False, "node_s": {},
                 "t0": time.perf_counter()}
        asyncio.run(respond_node(state, llm=_FakeLlm(), tts=tts,
                                 sessions=None, writer=events.append))
        kinds = [(e.node, e.kind) for e in events]
        self.assertNotIn(("llm", "thinking"), kinds)
        self.assertEqual(events[-1].data["reply"], "Hi there.")


class TestVoiceAgent(unittest.TestCase):
    def _run(self, ag, audio, **kw):
        import asyncio

        async def collect():
            return [e async for e in ag(audio, **kw)]

        return asyncio.run(collect())

    def test_event_order(self):
        import numpy as np

        ag = _agent_with([(0.0, 0.5)])
        events = self._run(ag, np.zeros(16000, dtype=np.float32), sr=16000)
        kinds = [(e.node, e.kind) for e in events]
        self.assertEqual(kinds[0], ("vad", "segments"))
        self.assertIn(("stt", "text"), kinds)
        self.assertIn(("llm", "token"), kinds)
        self.assertIn(("tts", "audio"), kinds)
        self.assertEqual(kinds[-1], ("turn", "summary"))
        summary = events[-1].data
        self.assertEqual(summary["text"], "hello")
        self.assertEqual(summary["reply"], "Hi there.")
        self.assertIn("vad", summary["node_s"])
        # llm done fires after the last token, before the summary
        llm_kinds = [k for n, k in kinds if n == "llm"]
        self.assertEqual(llm_kinds[-1], "done")

    def test_silence_short_circuits(self):
        import numpy as np

        ag = _agent_with([])
        events = self._run(ag, np.zeros(16000, dtype=np.float32), sr=16000)
        kinds = [(e.node, e.kind) for e in events]
        self.assertEqual(kinds[0], ("vad", "segments"))
        self.assertEqual(kinds[-1], ("turn", "summary"))
        self.assertNotIn("tts", [n for n, _ in kinds])
        self.assertEqual(events[-1].data["reply"], "")

    def test_session_memory(self):
        import numpy as np

        ag = _agent_with([(0.0, 0.5)])
        audio = np.zeros(16000, dtype=np.float32)
        self._run(ag, audio, sr=16000, session_id="s1")
        self._run(ag, audio, sr=16000, session_id="s1")
        hist = ag.sessions.history("s1")
        self.assertEqual(len(hist), 4)  # 2 turns x user+assistant

    def test_guards(self):
        import numpy as np

        ag = _agent_with([(0.0, 0.5)])
        with self.assertRaises(ValueError):
            self._run(ag, np.zeros(0, dtype=np.float32))
        big = np.zeros(61 * 16000, dtype=np.float32)
        with self.assertRaises(ValueError):
            self._run(ag, big)


if __name__ == "__main__":
    unittest.main()
