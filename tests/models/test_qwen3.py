"""Qwen3-leg tests: QK-norm + bias-free path on fake weights (CPU-fast).

No downloads, no GPU: a tiny 2-layer state dict in Qwen3 layout (q_norm /
k_norm present, no attention biases) proves the new code path runs, and a
Qwen2.5-layout dict (biases, no norms) proves the old path is intact.
"""
import unittest

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

VOCAB, HIDDEN, H, HK, DH, FF, LAYERS = 64, 32, 8, 2, 8, 64, 2
# NB: H*DH (64) != HIDDEN (32), like Qwen3-0.6B (2048 vs 1024): decode
# must not assume d == hidden.


def _fake_weights(qk_norm=True, biases=False):
    torch.manual_seed(0)
    w = {}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = torch.randn(HIDDEN)
        w[p + "self_attn.q_proj.weight"] = torch.randn(H * DH, HIDDEN) * 0.1
        w[p + "self_attn.k_proj.weight"] = torch.randn(HK * DH, HIDDEN) * 0.1
        w[p + "self_attn.v_proj.weight"] = torch.randn(HK * DH, HIDDEN) * 0.1
        w[p + "self_attn.o_proj.weight"] = torch.randn(HIDDEN, H * DH) * 0.1
        if biases:
            w[p + "self_attn.q_proj.bias"] = torch.randn(H * DH) * 0.1
            w[p + "self_attn.k_proj.bias"] = torch.randn(HK * DH) * 0.1
            w[p + "self_attn.v_proj.bias"] = torch.randn(HK * DH) * 0.1
        if qk_norm:
            w[p + "self_attn.q_norm.weight"] = torch.ones(DH)
            w[p + "self_attn.k_norm.weight"] = torch.ones(DH)
        w[p + "post_attention_layernorm.weight"] = torch.randn(HIDDEN)
        w[p + "mlp.gate_proj.weight"] = torch.randn(FF, HIDDEN) * 0.1
        w[p + "mlp.up_proj.weight"] = torch.randn(FF, HIDDEN) * 0.1
        w[p + "mlp.down_proj.weight"] = torch.randn(HIDDEN, FF) * 0.1
    w["model.norm.weight"] = torch.randn(HIDDEN)
    w["model.embed_tokens.weight"] = torch.randn(VOCAB, HIDDEN) * 0.1
    return w


def _engine(weights):
    from src.models.engines.qwen import QwenEngine
    from src.models.triton_kernels.qwen import build_cos_sin

    eng = QwenEngine.__new__(QwenEngine)
    eng.device = "cpu"
    eng.nlayers, eng.H, eng.Hk = LAYERS, H, HK
    eng.hidden, eng.dh = HIDDEN, DH
    import math

    eng.scale = 1 / math.sqrt(DH)
    eng.eps = 1e-6
    eng.w = weights
    eng.qk_norm = any(k.endswith("self_attn.q_norm.weight")
                      for k in weights)
    eng.max_len = 64
    eng.cos, eng.sin = build_cos_sin(64, DH, 1000000.0, "cpu")
    return eng


class TestQwen3Path(unittest.TestCase):
    def test_detect_and_generate(self):
        if not _HAS_TORCH:
            self.skipTest("torch missing")
        eng = _engine(_fake_weights(qk_norm=True, biases=False))
        self.assertTrue(eng.qk_norm)
        self.assertEqual(eng.dh, DH)
        r = eng.generate([1, 2, 3], max_new_tokens=4)
        self.assertLessEqual(len(r["ids"]), 4)
        self.assertGreaterEqual(len(r["ids"]), 0)
        streamed = [t for t, _ in eng.generate_stream([1, 2, 3],
                                                      max_new_tokens=4)]
        self.assertEqual(list(r["ids"]), streamed)

    def test_qwen25_path_intact(self):
        if not _HAS_TORCH:
            self.skipTest("torch missing")
        eng = _engine(_fake_weights(qk_norm=False, biases=True))
        self.assertFalse(eng.qk_norm)
        r = eng.generate([1, 2, 3], max_new_tokens=4)
        self.assertLessEqual(len(r["ids"]), 4)

    def test_stop_ids_shared(self):
        from src.models.engines.qwen import QwenEngine

        self.assertIn(151645, QwenEngine.STOP_IDS)
        self.assertIn(151643, QwenEngine.STOP_IDS)

    def test_fallback_dims_are_qwen3(self):
        from src.models.engines.qwen import FALLBACK_DIMS, MODEL_ID

        self.assertEqual(MODEL_ID, "Qwen/Qwen3-0.6B")
        self.assertEqual(FALLBACK_DIMS["num_hidden_layers"], 28)
        self.assertEqual(FALLBACK_DIMS["head_dim"], 128)
        self.assertEqual(FALLBACK_DIMS["num_key_value_heads"], 8)


class TestThinking(unittest.TestCase):
    def test_decode_preserves_think(self):
        """decode() must NOT strip <think>: callers split via split_thinking.

        Stripping here destroyed the terminal think→toolcall loop
        (decode_with_thinking could never recover thinking once stripped).
        """
        import asyncio

        from src.models.llm import LlmConfig, LlmModel, split_thinking

        model = LlmModel(LlmConfig(thinking=True))

        class StubTok:
            def decode(self, ids, skip_special_tokens=True):
                return "<think>hmm, reasoning</think>Hi there."

        model._tok = StubTok()
        text = asyncio.run(model.decode([1, 2, 3]))
        self.assertEqual(text, "<think>hmm, reasoning</think>Hi there.")
        th, ans = split_thinking(text)
        self.assertEqual((th, ans), ("hmm, reasoning", "Hi there."))
        th2, ans2 = asyncio.run(model.decode_with_thinking([1, 2, 3]))
        self.assertEqual((th2, ans2), ("hmm, reasoning", "Hi there."))

    def test_off_by_default(self):
        from src.models.llm import LlmConfig

        self.assertFalse(LlmConfig().thinking)
        self.assertEqual(LlmConfig().model, "openbmb/MiniCPM5-1B")


class TestQwen3Defaults(unittest.TestCase):
    def test_agent_default(self):
        from src.main import VoiceAgentConfig

        self.assertEqual(VoiceAgentConfig().llm.model, "openbmb/MiniCPM5-1B")

    def test_no_yaml_configs_dir(self):
        import os

        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        self.assertFalse(os.path.exists(os.path.join(root, "configs")),
                         "configs/ is dead: dataclasses replaced YAML")


if __name__ == "__main__":
    unittest.main()
