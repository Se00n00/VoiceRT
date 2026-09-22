"""Unified VoiceAgent tests: config, voice turns, injection, memory.

No weights, no GPU: legs are faked; the deep-agent loop runs for real
through LocalChatModel with scripted raw outputs.
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
        self.assertEqual(cfg.sessions_dir, "sessions")
        self.assertEqual(cfg.max_agent_steps, 6)
        self.assertIsNone(cfg.sandbox)
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

    def test_chat_model_wraps_local_llm(self):
        from src.agent.chat_model import LocalChatModel
        from src.main import VoiceAgent

        ag = VoiceAgent()
        self.assertIsInstance(ag.chat_model, LocalChatModel)
        self.assertIs(ag.chat_model.llm, ag.llm)


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


def _chips(raw):
    from src.models.llm import LlmToken

    words = raw.split(" ") or [raw]
    return [LlmToken(token_id=100 + i, piece=(w + " "), first=(i == 0))
            for i, w in enumerate(words)]


class _FakeLlm:
    """Scripted leg with streaming (chips) like the real LlmModel."""

    def __init__(self, raws):
        self.raws = list(raws)
        self._full = ""

    def messages_for_terminal(self, text, history=None, cwd="", observation=""):
        return [{"role": "user", "content": text}]

    async def generate(self, messages, max_tokens=None, tools=None, stop=None):
        from src.models.llm import LlmResult

        self.last_tools = tools
        return LlmResult(text=self.raws.pop(0) if self.raws else "done chatting.")

    async def stream(self, messages, max_tokens=None, tools=None, stop=None):
        raw = self.raws.pop(0) if self.raws else "done chatting."
        self._full = raw
        for tok in _chips(raw):
            yield tok

    async def decode(self, ids):
        return self._full


class _FakeTts:
    def __init__(self):
        import numpy as np

        self._wav = np.zeros(240, dtype=np.float32)
        self.spoken = []

    async def speak(self, text):
        from src.models.tts import TtsAudio

        self.spoken.append(text)
        return TtsAudio(wav=self._wav, sample_rate=24000, sentence=text,
                        synth_s=0.01)


def _agent_with(segs, raws, **kw):
    import tempfile

    from src.agent.chat_model import LocalChatModel
    from src.main import VoiceAgent, VoiceAgentConfig

    cfg = {"sessions_dir": tempfile.mkdtemp(prefix="vagent-")}
    cfg.update(kw)
    ag = VoiceAgent(VoiceAgentConfig(**cfg))
    llm = _FakeLlm(raws)
    ag.vad, ag.stt, ag.llm, ag.tts = (_FakeVad(segs), _FakeStt(),
                                      llm, _FakeTts())
    ag.chat_model = LocalChatModel(llm=llm)
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

    def test_leg_device_reads_backend_attr(self):
        from src.models.llm import leg_device

        class FakeCuda:
            _leg = type("Leg", (), {"device": "cuda:0"})()

        class FakeCpu:
            _leg = type("Leg", (), {"device": "cpu"})()

        self.assertEqual(leg_device(FakeCuda()), "cuda:0")
        self.assertEqual(leg_device(FakeCpu()), "cpu")
        self.assertEqual(leg_device(object()), "unknown")

    def test_assert_cuda_leg_aborts_off_cuda(self):
        from src.models.llm import assert_cuda_leg

        class FakeCuda:
            _leg = type("Leg", (), {"device": "cuda:0"})()

        class FakeCpu:
            _leg = type("Leg", (), {"device": "cpu"})()

        self.assertEqual(assert_cuda_leg(FakeCuda()), "cuda:0")
        with self.assertRaises(SystemExit) as cm:
            assert_cuda_leg(FakeCpu())
        self.assertIn("CUDA", str(cm.exception))

    def test_voice_thinking_not_spoken(self):
        import asyncio
        import numpy as np

        ag = _agent_with([(0.0, 0.5)],
                         ["<think>choose greeting</think>Hi there."])
        tts = ag.tts
        events = asyncio.run(_collect_call(
            ag, np.zeros(16000, dtype=np.float32), sr=16000))
        kinds = [(e.node, e.kind) for e in events]
        self.assertIn(("llm", "thinking"), kinds)
        think = [e for e in events if e.kind == "thinking"][0]
        self.assertIn("choose greeting", think.data["text"])
        # thoughts never reach TTS; the answer does, whole
        self.assertTrue(tts.spoken)
        self.assertFalse(any("choose greeting" in s for s in tts.spoken))
        self.assertEqual("".join(tts.spoken), "Hi there.")
        self.assertEqual(events[-1].data["reply"], "Hi there.")

    def test_voice_no_think_unchanged(self):
        import asyncio
        import numpy as np

        ag = _agent_with([(0.0, 0.5)], ["Hi there."])
        events = asyncio.run(_collect_call(
            ag, np.zeros(16000, dtype=np.float32), sr=16000))
        kinds = [(e.node, e.kind) for e in events]
        self.assertNotIn(("llm", "thinking"), kinds)
        self.assertEqual(events[-1].data["reply"], "Hi there.")


async def _collect_call(ag, audio, **kw):
    return [e async for e in ag(audio, **kw)]


class TestVoiceAgent(unittest.TestCase):
    def _run(self, ag, audio, **kw):
        import asyncio

        return asyncio.run(_collect_call(ag, audio, **kw))

    def test_event_order(self):
        import numpy as np

        ag = _agent_with([(0.0, 0.5)], ["Hi there."])
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

    def test_voice_tool_call_runs(self):
        import numpy as np

        ag = _agent_with([(0.0, 0.5)],
                         ['{"action": "exec", "command": "echo hi"}',
                          '{"action": "done", "reply": "Did it."}'])
        events = self._run(ag, np.zeros(16000, dtype=np.float32), sr=16000)
        kinds = [(e.node, e.kind) for e in events]
        self.assertIn(("term", "action"), kinds)
        self.assertIn(("term", "observation"), kinds)
        obs = [e for e in events if e.kind == "observation"][0]
        self.assertIn("hi", obs.data["observation"])
        self.assertEqual(events[-1].data["reply"], "Did it.")
        self.assertEqual("".join(ag.tts.spoken), "Did it.")

    def test_silence_short_circuits(self):
        import numpy as np

        ag = _agent_with([], [])
        events = self._run(ag, np.zeros(16000, dtype=np.float32), sr=16000)
        kinds = [(e.node, e.kind) for e in events]
        self.assertEqual(kinds[0], ("vad", "segments"))
        self.assertEqual(kinds[-1], ("turn", "summary"))
        self.assertNotIn("tts", [n for n, _ in kinds])
        self.assertEqual(events[-1].data["reply"], "")

    def test_session_memory(self):
        import numpy as np

        ag = _agent_with([(0.0, 0.5)], ["Hi there."])
        audio = np.zeros(16000, dtype=np.float32)
        self._run(ag, audio, sr=16000, session_id="s1")
        self._run(ag, audio, sr=16000, session_id="s1")
        hist = ag.sessions.history("s1")
        self.assertEqual(len(hist), 4)  # 2 turns x user+assistant

    def test_guards(self):
        import numpy as np

        ag = _agent_with([(0.0, 0.5)], ["Hi there."])
        with self.assertRaises(ValueError):
            self._run(ag, np.zeros(0, dtype=np.float32))
        big = np.zeros(61 * 16000, dtype=np.float32)
        with self.assertRaises(ValueError):
            self._run(ag, big)

    def test_injected_message_queued_then_drains(self):
        import asyncio
        import numpy as np

        gate = asyncio.Event()

        class GateLlm:
            """Generate-only leg (no stream): first call blocks on gate."""

            def __init__(self, raws):
                self.raws = list(raws)
                self.calls = 0

            def messages_for_terminal(self, text, history=None, cwd="",
                                      observation=""):
                return [{"role": "user", "content": text}]

            async def generate(self, messages, max_tokens=None,
                               tools=None, stop=None):
                from src.models.llm import LlmResult

                self.calls += 1
                if self.calls == 1:
                    await gate.wait()
                return LlmResult(
                    text=self.raws.pop(0) if self.raws else "done chatting.")

            async def decode(self, ids):
                return ""

        ag = _agent_with([(0.0, 0.5)], ["working on it", "second done"])
        ag.llm = GateLlm(["working on it", "second done"])
        from src.agent.chat_model import LocalChatModel

        ag.chat_model = LocalChatModel(llm=ag.llm)
        audio = np.zeros(16000, dtype=np.float32)

        async def go():
            t1 = asyncio.create_task(_collect_call(ag, audio, sr=16000,
                                                  session_id="s9"))
            for _ in range(200):
                if ag.llm.calls >= 1:
                    break
                await asyncio.sleep(0.01)
            second = await _collect_call(ag, audio, sr=16000,
                                         session_id="s9")
            gate.set()
            first = await t1
            return first, second

        first, second = asyncio.run(go())
        # contender runs VAD/STT, then sees exactly one queued event and an
        # empty summary (its text was injected into the active turn)
        skinds = [(e.node, e.kind) for e in second]
        self.assertIn(("term", "queued"), skinds)
        self.assertNotIn(("llm", "done"), skinds)
        self.assertEqual(second[-1].data["reply"], "")
        # the active turn drains the injection: both replies surface in order
        chats = [e.data["text"] for e in first
                 if e.node == "llm" and e.kind == "done"]
        self.assertEqual(chats, ["working on it", "second done"])
        self.assertEqual(first[-1].data["reply"], "second done")
        self.assertEqual(len(ag.sessions.history("s9")), 4)


if __name__ == "__main__":
    unittest.main()
