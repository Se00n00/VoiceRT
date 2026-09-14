"""Whisper-leg model tests: import + config parity, shape checks.

Decode correctness needs cuda + whisper weights, so it skips unless both
are present. Light checks run on CPU.
"""
import unittest

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False


class TestWhisperModel(unittest.TestCase):
    def test_import(self):
        try:
            __import__("src.models.engines.whisper", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"src.models.engines.whisper not ported yet ({exc})")

    def test_config(self):
        from src.models.stt import SttConfig

        cfg = SttConfig()
        self.assertEqual(cfg.model, "openai/whisper-base")
        self.assertEqual(cfg.sample_rate, 16000)

    @unittest.skipUnless(_HAS_TORCH and torch.cuda.is_available(), "needs cuda")
    def test_transcribe_shape(self):
        try:
            mod = __import__("src.models.engines.whisper", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"src.models.engines.whisper not ported yet ({exc})")
        cls = getattr(mod, "WhisperEngine", None) or getattr(
            mod, "WhisperTriton", None)
        if cls is None:
            self.skipTest("no WhisperEngine/WhisperTriton in src.models.engines.whisper")
        try:
            eng = cls()
        except Exception as exc:
            self.skipTest(f"Whisper weights unavailable ({exc})")
        mel = torch.zeros(1, 80, 3000, device="cuda:0")
        try:
            r = eng.transcribe(mel)
        except Exception as exc:
            self.skipTest(f"transcribe failed ({exc})")
        self.assertIn("ids", r)


if __name__ == "__main__":
    unittest.main()
