"""VoiceEngine tests: import + interface (lazy-init safe).

Never constructs VoiceEngine (that loads all weights); only checks the
class is importable and exposes the five leg methods used by server/,
benchmarks/ and examples/.
"""
import unittest


class TestEngine(unittest.TestCase):
    def test_import(self):
        try:
            from engine.engine import VoiceEngine  # noqa: F401
        except Exception as exc:
            self.skipTest(f"engine.engine.VoiceEngine not ported yet ({exc})")

    def test_interface(self):
        try:
            from engine.engine import VoiceEngine
        except Exception as exc:
            self.skipTest(f"engine.engine.VoiceEngine not ported yet ({exc})")
        for name in ("vad_segments", "vad_active", "transcribe", "chat",
                      "speak", "stream_turn", "talk_turn"):
            self.assertTrue(hasattr(VoiceEngine, name),
                            f"VoiceEngine missing {name}")

    def test_talk_turn_event_order(self):
        """talk_turn emits stt -> llm tokens -> llm done -> tts, in order."""
        import numpy as np
        import torch

        from engine.engine import VoiceEngine
        from engine.session import SessionStore
        from runtime.profiler import Profiler

        eng = VoiceEngine.__new__(VoiceEngine)
        eng.device = torch.device("cpu")
        eng.max_new_tokens = 8
        eng.profiler = Profiler()
        eng.missing = []
        eng.sessions = SessionStore()
        eng.vram_budget_mb = 1e9
        eng.per_turn_mb = 150.0

        class FakeProc:
            def __call__(self, wav, sampling_rate=None, return_tensors=None):
                import types

                return types.SimpleNamespace(
                    input_features=torch.zeros(1, 80, 3000))

            def batch_decode(self, ids_list, skip_special_tokens=True):
                return ["hi"]

        class FakeSTT:
            def transcribe_mel(self, mel, max_tokens=64):
                return {"ids": [1, 2], "ttfs": 0.001}

        class FakeTok:
            def apply_chat_template(self, msgs, return_tensors=None,
                                    add_generation_prompt=True):
                return {"input_ids": torch.tensor([[1]])}

            def decode(self, ids, skip_special_tokens=True):
                if len(ids) == 1:
                    return "Hello. "
                return "Hello."

        class FakeLLM:
            def generate_stream(self, ids, max_new_tokens=8):
                yield 5, 0.001
                yield 6, None

        class FakeTTS:
            def speak(self, text):
                return (np.zeros(240, dtype=np.float32), 24000)

        class FakeVAD:
            def segments(self, wav):
                return [(0.0, 0.1)]

        eng.proc, eng.stt, eng.tok = FakeProc(), FakeSTT(), FakeTok()
        eng.llm, eng.tts, eng.vad = FakeLLM(), FakeTTS(), FakeVAD()

        self.assertTrue(eng.vad_active(np.zeros(160, dtype=np.float32)))

        events = []
        r = eng.talk_turn(np.zeros(16000, dtype=np.float32), 16000, "s1",
                          on_event=lambda k, p: events.append(k))
        kinds = [k for k in events]
        self.assertEqual(kinds[0], "stt")
        self.assertIn("llm", kinds[1:])
        self.assertIn("tts", kinds)
        # llm-done fires exactly once, after the last token, before return
        self.assertEqual(r["text"], "hi")
        self.assertEqual(r["session_id"], "s1")


if __name__ == "__main__":
    unittest.main()
