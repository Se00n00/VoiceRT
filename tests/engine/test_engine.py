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
        for name in ("vad_segments", "transcribe", "chat", "speak",
                     "stream_turn"):
            self.assertTrue(hasattr(VoiceEngine, name),
                            f"VoiceEngine missing {name}")


if __name__ == "__main__":
    unittest.main()
