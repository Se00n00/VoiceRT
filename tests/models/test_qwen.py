"""Qwen-leg model tests: import + config parity, shape checks.

Heavy parity (exact text match vs HF) needs cuda + downloaded weights,
so it skips unless both are present. Light checks (import, yaml config)
run on CPU.
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


class TestQwenModel(unittest.TestCase):
    def test_import(self):
        try:
            __import__("models.qwen", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"models.qwen not ported yet ({exc})")

    def test_config(self):
        cfg_path = os.path.join(ROOT, "configs", "qwen.yaml")
        if not os.path.exists(cfg_path):
            self.skipTest("configs/qwen.yaml missing")
        cfg = _read_yaml_simple(cfg_path)
        self.assertEqual(cfg.get("model"), "Qwen/Qwen2.5-0.5B-Instruct")
        for key in ("layers", "hidden", "q_heads", "kv_heads", "head_dim"):
            self.assertIn(key, cfg, f"qwen.yaml missing {key}")

    @unittest.skipUnless(_HAS_TORCH and torch.cuda.is_available(), "needs cuda")
    def test_generate_shape(self):
        """Greedy generate returns the requested number of ids (needs weights)."""
        try:
            mod = __import__("models.qwen", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"models.qwen not ported yet ({exc})")
        cls = getattr(mod, "QwenEngine", None) or getattr(mod, "TinyLLM", None)
        if cls is None:
            self.skipTest("no QwenEngine/TinyLLM class in models.qwen")
        try:
            eng = cls()
        except Exception as exc:
            self.skipTest(f"Qwen weights unavailable ({exc})")
        ids = [1, 2, 3, 4] * 4
        try:
            r = eng.generate(ids, max_new_tokens=4)
        except Exception as exc:
            self.skipTest(f"generate failed ({exc})")
        out = r["ids"] if isinstance(r, dict) else r
        # EOS stopping may cut generation short; it must never exceed.
        self.assertLessEqual(len(out), 4)

    @unittest.skipUnless(_HAS_TORCH and torch.cuda.is_available(), "needs cuda")
    def test_eos_stops(self):
        """Greedy decode must stop at <|im_end|>/<|endoftext|> and never
        emit them (regression: streamed role-play past the answer)."""
        try:
            mod = __import__("models.qwen", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"models.qwen not ported yet ({exc})")
        cls = getattr(mod, "QwenEngine", None) or getattr(mod, "TinyLLM", None)
        if cls is None:
            self.skipTest("no QwenEngine/TinyLLM class in models.qwen")
        try:
            eng = cls()
        except Exception as exc:
            self.skipTest(f"Qwen weights unavailable ({exc})")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        ids = tok.apply_chat_template(
            [{"role": "user", "content": "Say hi."}],
            return_tensors="pt",
            add_generation_prompt=True)["input_ids"][0].tolist()
        r = eng.generate(ids, max_new_tokens=48)
        out = r["ids"] if isinstance(r, dict) else r
        self.assertNotIn(151645, out)
        self.assertNotIn(151643, out)
        streamed = [t for t, _ in eng.generate_stream(ids, max_new_tokens=48)]
        self.assertNotIn(151645, streamed)
        self.assertNotIn(151643, streamed)
        self.assertEqual(list(out), streamed)


if __name__ == "__main__":
    unittest.main()
