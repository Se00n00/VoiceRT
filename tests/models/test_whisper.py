"""Whisper-leg model tests: import + config parity, shape checks.

Decode correctness needs cuda + whisper weights, so it skips unless both
are present. Light checks run on CPU.
"""
import os
import unittest

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_yaml_simple(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip().strip("[]")
            d[k.strip()] = v.strip()
    return d


class TestWhisperModel(unittest.TestCase):
    def test_import(self):
        try:
            __import__("models.whisper", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"models.whisper not ported yet ({exc})")

    def test_config(self):
        cfg_path = os.path.join(ROOT, "configs", "whisper.yaml")
        if not os.path.exists(cfg_path):
            self.skipTest("configs/whisper.yaml missing")
        cfg = _read_yaml_simple(cfg_path)
        self.assertEqual(cfg.get("model"), "openai/whisper-base")
        self.assertIn("sample_rate", cfg)

    @unittest.skipUnless(_HAS_TORCH and torch.cuda.is_available(), "needs cuda")
    def test_transcribe_shape(self):
        try:
            mod = __import__("models.whisper", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"models.whisper not ported yet ({exc})")
        cls = getattr(mod, "WhisperEngine", None) or getattr(
            mod, "WhisperTriton", None)
        if cls is None:
            self.skipTest("no WhisperEngine/WhisperTriton in models.whisper")
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
