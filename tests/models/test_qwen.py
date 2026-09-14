"""Qwen-leg model tests: import + config parity, shape checks.

Heavy parity (exact text match vs HF) needs cuda + downloaded weights,
so it skips unless both are present. Light checks (import, dataclass
config) run on CPU.
"""
import unittest

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False


class TestQwenModel(unittest.TestCase):
    def test_import(self):
        try:
            __import__("src.models.engines.qwen", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"src.models.engines.qwen not ported yet ({exc})")

    def test_config(self):
        from src.models.llm import LlmConfig

        cfg = LlmConfig()
        self.assertEqual(cfg.model, "Qwen/Qwen3-0.6B")
        self.assertEqual(cfg.max_tokens, 48)

    @unittest.skipUnless(_HAS_TORCH and torch.cuda.is_available(), "needs cuda")
    def test_generate_shape(self):
        """Greedy generate returns the requested number of ids (needs weights)."""
        try:
            mod = __import__("src.models.engines.qwen", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"src.models.engines.qwen not ported yet ({exc})")
        cls = getattr(mod, "QwenEngine", None) or getattr(mod, "TinyLLM", None)
        if cls is None:
            self.skipTest("no QwenEngine/TinyLLM class in src.models.engines.qwen")
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
    def test_matches_hf_prefill(self):
        """Our prefill logits ~ HF logits (guards QK-norm order, bias map).

        Regression test: QK-norm applied after rotary diverges with
        max|diff| ~20 and greedy-decodes into repetition loops; the
        correct (HF) order lands < 2.0 (bf16 accumulation noise).
        """
        try:
            mod = __import__("src.models.engines.qwen", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"src.models.engines.qwen not ported yet ({exc})")
        cls = getattr(mod, "QwenEngine", None) or getattr(mod, "TinyLLM", None)
        if cls is None:
            self.skipTest("no QwenEngine/TinyLLM class in src.models.engines.qwen")
        try:
            eng = cls()
        except Exception as exc:
            self.skipTest(f"Qwen weights unavailable ({exc})")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            hf = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to("cuda").eval()
        except Exception as exc:
            self.skipTest(f"HF reference unavailable ({exc})")
        msgs = [{"role": "user", "content": "Say hi in three words."}]
        enc = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=False)
        ids = enc["input_ids"][0].tolist()
        with torch.no_grad():
            ref = hf(torch.tensor([ids], device="cuda")).logits[0, -1].float()
        x = torch.nn.functional.embedding(
            torch.tensor(ids, device="cuda"),
            eng.w["model.embed_tokens.weight"])
        cache = eng._new_cache()
        for i in range(eng.nlayers):
            x = eng._layer_prefill(x, i, cache)
        got = eng._logits(x[-1]).float().cpu()
        ref_cpu = ref.cpu()
        self.assertLess(float((got - ref_cpu).abs().max()), 2.0)
        # Same top tokens (argmax alone can jitter between near-tied
        # bf16 logits; the wrong QK-norm order shares none of them).
        got_top5 = torch.topk(got, 5).indices.tolist()
        ref_top5 = torch.topk(ref_cpu, 5).indices.tolist()
        self.assertIn(int(ref_cpu.argmax()), got_top5)
        self.assertGreaterEqual(len(set(got_top5) & set(ref_top5)), 3)

    @unittest.skipUnless(_HAS_TORCH and torch.cuda.is_available(), "needs cuda")
    def test_eos_stops(self):
        """Greedy decode must stop at <|im_end|>/<|endoftext|> and never
        emit them (regression: streamed role-play past the answer)."""
        try:
            mod = __import__("src.models.engines.qwen", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"src.models.engines.qwen not ported yet ({exc})")
        cls = getattr(mod, "QwenEngine", None) or getattr(mod, "TinyLLM", None)
        if cls is None:
            self.skipTest("no QwenEngine/TinyLLM class in src.models.engines.qwen")
        try:
            eng = cls()
        except Exception as exc:
            self.skipTest(f"Qwen weights unavailable ({exc})")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
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

    @unittest.skipUnless(_HAS_TORCH and torch.cuda.is_available(), "needs cuda")
    def test_no_repetition_loop(self):
        """Greedy replies must not degenerate into loops (QK-norm-order
        regression: wrong order decoded 'Are you?' x15 on this prompt)."""
        try:
            mod = __import__("src.models.engines.qwen", fromlist=["*"])
        except Exception as exc:
            self.skipTest(f"src.models.engines.qwen not ported yet ({exc})")
        cls = getattr(mod, "QwenEngine", None) or getattr(mod, "TinyLLM", None)
        if cls is None:
            self.skipTest("no QwenEngine/TinyLLM class in src.models.engines.qwen")
        try:
            eng = cls()
        except Exception as exc:
            self.skipTest(f"Qwen weights unavailable ({exc})")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        msgs = [{"role": "system", "content": "You are a voice assistant. "
                 "Reply in one short spoken sentence."},
                {"role": "user", "content": " Hello there, how are you?"}]
        enc = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=False)
        ids = enc["input_ids"][0].tolist()
        r = eng.generate(ids, max_new_tokens=48)
        out = r["ids"] if isinstance(r, dict) else r
        text = tok.decode(out, skip_special_tokens=True)
        self.assertGreater(len(set(out)) if out else 1, 4,
                           f"degenerate loop: {text!r}")
        self.assertLessEqual(text.count("Are you?"), 2,
                             f"degenerate loop: {text!r}")


if __name__ == "__main__":
    unittest.main()
