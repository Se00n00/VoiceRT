"""Model runner — executes the batched forward with paged KV.

Real Qwen weights only. No dummy / random-weight fallback.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

from src.inference.batching import InputBatch
from src.inference.config import EngineConfig
from src.inference.kv_cache import PagedKVCache
from src.models.runtime.device import select_device
from src.models.triton_kernels.qwen import (
    rmsnorm,
    rope,
    rope_batched,
    swiglu,
    gqa_decode_attn,
    build_cos_sin,
)
try:
    from src.models.triton_kernels.paged_attention import paged_gqa_decode, paged_gqa_prefill
    _HAS_PAGED_ATTENTION = True
except Exception:
    _HAS_PAGED_ATTENTION = False

__all__ = ["ModelRunner", "QwenRunner", "create_runner"]


class ModelRunner:
    """Interface."""

    def forward(self, batch: InputBatch, kv_cache: PagedKVCache) -> torch.Tensor:
        """Return next-token logits [num_seqs, vocab_size]."""
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# QwenRunner — batched paged forward over real Qwen weights (only runner)
# ---------------------------------------------------------------------------

class QwenRunner(ModelRunner):
    """Batched Qwen runner with paged KV — real weights only, fail loud.

    Loads weights via src/models/engines/qwen (same as QwenEngine) and
    executes batched paged attention (gather from PagedKVCache).
    No dummy fallback: missing weights / OOM raise.
    """

    def __init__(self, config: EngineConfig, device: str | torch.device = "cpu"):
        self.cfg = config
        if isinstance(device, str):
            device = select_device(device)
            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
        self.device = device
        self.loaded = False
        self._weights = None
        self._qwen_cfg = None
        self.cos = None
        self.sin = None
        self.eps = 1e-6
        self.H = 16
        self.Hk = 8
        self.dh = 128
        self.hidden = 1024
        self.nlayers = 28
        self._try_load()

    def _try_load(self):
        from src.models.engines.qwen import load_hf_config, load_weights

        cfg = load_hf_config(self.cfg.model)
        self._qwen_cfg = cfg
        self.H = cfg["num_attention_heads"]
        self.Hk = cfg["num_key_value_heads"]
        self.hidden = cfg["hidden_size"]
        self.dh = int(cfg.get("head_dim") or self.hidden // self.H)
        self.eps = float(cfg["rms_norm_eps"])
        self.nlayers = int(cfg["num_hidden_layers"])
        # real weights — raise on missing / OOM, never fallback
        w = load_weights(device=str(self.device), repo_id=self.cfg.model)
        self._weights = w
        n = min(2048, cfg["max_position_embeddings"])
        theta = float(cfg.get("rope_theta") or 1000000.0)
        self.cos, self.sin = build_cos_sin(n, self.dh, theta, str(self.device))
        # vocab size from lm_head
        if "lm_head.weight" in w:
            self._vocab = w["lm_head.weight"].shape[0]
        else:
            self._vocab = w["model.embed_tokens.weight"].shape[0]
        self.loaded = True

    def vocab_size(self) -> int:
        return int(self._vocab)

    def _build_block_tables(self, batch, kv_cache, seq_groups):
        """Build block_tables tensor for paged attention kernels.
        Returns: block_tables [B, max_blocks], seq_lens [B], chunk_starts [B], chunk_lens [B]"""
        B = len(seq_groups)
        max_blocks = kv_cache.config.num_blocks
        block_tables = torch.full((B, 64), -1, dtype=torch.int32, device=self.device)
        seq_lens = torch.zeros(B, dtype=torch.int32, device=self.device)
        chunk_starts = torch.zeros(B, dtype=torch.int32, device=self.device)
        chunk_lens = torch.zeros(B, dtype=torch.int32, device=self.device)
        for i, sg in enumerate(seq_groups):
            seq = sg.seq
            table = kv_cache.block_manager._tables.get(seq.seq_id, [])
            seq_len = seq.num_tokens
            for j, bid in enumerate(table):
                if j < 64:
                    block_tables[i, j] = bid
            seq_lens[i] = seq_len
            if batch.is_prefill[i]:
                chunk_starts[i] = seq.num_computed_tokens
                chunk_lens[i] = batch.num_tokens_per_seq[i]
            else:
                chunk_lens[i] = 1
        return block_tables, seq_lens, chunk_starts, chunk_lens

    def forward(self, batch: InputBatch, kv_cache: PagedKVCache) -> torch.Tensor:
        # real Qwen batched paged forward — no dummy path
        from src.models.engines.qwen import lm_head_weight

        # Try fused paged attention if available
        use_paged = _HAS_PAGED_ATTENTION and batch.num_seqs > 1 and batch.is_cuda
        if use_paged:
            block_tables, seq_lens, chunk_starts, chunk_lens = self._build_block_tables(batch, kv_cache, batch.seq_groups)

        seq_logits = []
        offset = 0
        for idx, sg in enumerate(batch.seq_groups):
            seq = sg.seq
            n_tok = batch.num_tokens_per_seq[idx]
            is_prefill = batch.is_prefill[idx]
            tokens = batch.input_ids[offset : offset + n_tok]
            positions = batch.positions[offset : offset + n_tok]
            offset += n_tok

            # embed
            w = self._weights
            x = F.embedding(tokens.to(self.device), w["model.embed_tokens.weight"])
            # x: [L, hidden] or [1, hidden]
            if is_prefill:
                L = x.shape[0]
                pos0 = int(seq.num_computed_tokens)
                for layer in range(self.nlayers):
                    p = f"model.layers.{layer}."
                    h = rmsnorm(x, w[p + "input_layernorm.weight"], self.eps)
                    # --- prefill attention ---
                    # proj
                    q = F.linear(h, w[p + "self_attn.q_proj.weight"], w.get(p + "self_attn.q_proj.bias"))
                    k = F.linear(h, w[p + "self_attn.k_proj.weight"], w.get(p + "self_attn.k_proj.bias"))
                    v = F.linear(h, w[p + "self_attn.v_proj.weight"], w.get(p + "self_attn.v_proj.bias"))
                    # QK-norm if present (Qwen3)
                    qw = w.get(p + "self_attn.q_norm.weight")
                    kw = w.get(p + "self_attn.k_norm.weight")
                    if qw is not None and kw is not None:
                        q = rmsnorm(q.view(L, self.H, self.dh), qw).view(L, -1)
                        k = rmsnorm(k.view(L, self.Hk, self.dh), kw).view(L, -1)
                    # RoPE batched
                    q = rope_batched(q.view(L, self.H, self.dh), self.cos, self.sin, pos0).view(L, -1)
                    k = rope_batched(k.view(L, self.Hk, self.dh), self.cos, self.sin, pos0).view(L, -1)
                    # store into paged cache: need [L, Hk, dh]
                    k_paged = k.view(L, self.Hk, self.dh)
                    v_paged = v.view(L, self.Hk, self.dh)
                    kv_cache.store_prefill(seq.seq_id, layer, k_paged, v_paged)
                    # now gather for causal attention over total length
                    total = pos0 + L
                    Kg, Vg = kv_cache.gather(seq.seq_id, layer, total)
                    # compute attention for chunk using SDPA per position
                    # Use paged attention if available
                    if use_paged and _HAS_PAGED_ATTENTION:
                        # Build per-seq input for paged prefill
                        # For single seq in batch, call paged_gqa_prefill
                        scale = 1.0 / math.sqrt(self.dh)
                        q_proj = q.view(L, self.H, self.dh)
                        # Build inputs for this seq
                        seq_q = q_proj.unsqueeze(0)  # [1, L, H, dh] -> need [1, H, dh] per token
                        # For prefill we need to process each position; keep loop for now
                        # TODO: replace with paged_gqa_prefill when chunk_starts/chunk_lens support per-pos
                    # compute attention for chunk using SDPA per position
                    scale = 1.0 / math.sqrt(self.dh)
                    outs = []
                    q_3d = q.view(L, self.H, self.dh)
                    Kg_3d = Kg  # [total, Hk, dh]
                    Vg_3d = Vg
                    for i in range(L):
                        qi = q_3d[i]  # [H, dh]
                        K_slice = Kg_3d[: pos0 + i + 1].permute(1, 0, 2)  # [Hk, N, dh]
                        V_slice = Vg_3d[: pos0 + i + 1].permute(1, 0, 2)
                        oi = gqa_decode_attn(qi, K_slice, V_slice, scale)
                        outs.append(oi)
                    o = torch.stack(outs, dim=0).reshape(L, -1)
                    o = F.linear(o, w[p + "self_attn.o_proj.weight"], None)
                    x = x + o
                    # MLP
                    h2 = rmsnorm(x, w[p + "post_attention_layernorm.weight"], self.eps)
                    g = F.linear(h2, w[p + "mlp.gate_proj.weight"], None)
                    u = F.linear(h2, w[p + "mlp.up_proj.weight"], None)
                    mlp = F.linear(swiglu(g, u), w[p + "mlp.down_proj.weight"], None)
                    x = x + mlp
                # logits from last token
                h = rmsnorm(x[-1:].reshape(1, -1), w["model.norm.weight"], self.eps)
                logits = F.linear(h.reshape(-1), lm_head_weight(w))
                seq_logits.append(logits)
            else:
                # decode single token
                pos = int(positions[0].item())
                # x is [1, hidden]
                for layer in range(self.nlayers):
                    p = f"model.layers.{layer}."
                    h = rmsnorm(x, w[p + "input_layernorm.weight"], self.eps)
                    # decode step
                    q = F.linear(h[0], w[p + "self_attn.q_proj.weight"], w.get(p + "self_attn.q_proj.bias")).view(self.H, self.dh)
                    k1 = F.linear(h[0], w[p + "self_attn.k_proj.weight"], w.get(p + "self_attn.k_proj.bias")).view(self.Hk, self.dh)
                    v1 = F.linear(h[0], w[p + "self_attn.v_proj.weight"], w.get(p + "self_attn.v_proj.bias")).view(self.Hk, self.dh)
                    qw = w.get(p + "self_attn.q_norm.weight")
                    kw = w.get(p + "self_attn.k_norm.weight")
                    if qw is not None and kw is not None:
                        q = rmsnorm(q, qw)
                        k1 = rmsnorm(k1, kw)
                    q = rope(q, self.cos, self.sin, pos)
                    k1 = rope(k1, self.cos, self.sin, pos)
                    kv_cache.store_decode(seq.seq_id, layer, pos, k1, v1)
                    K, V = kv_cache.gather(seq.seq_id, layer, pos + 1)
                    K = K.permute(1, 0, 2)
                    V = V.permute(1, 0, 2)
                    scale = 1.0 / math.sqrt(self.dh)
                    # Use paged attention for batched decode if available
                    if use_paged and _HAS_PAGED_ATTENTION and batch.num_seqs > 1:
                        # Will be handled in batch mode; for single seq use existing
                        pass
                    K, V = kv_cache.gather(seq.seq_id, layer, pos + 1)
                    K = K.permute(1, 0, 2)
                    V = V.permute(1, 0, 2)
                    scale = 1.0 / math.sqrt(self.dh)
                    o = gqa_decode_attn(q, K, V, scale)
                    o = F.linear(o.reshape(-1), w[p + "self_attn.o_proj.weight"], None).unsqueeze(0)
                    x = x + o
                    h2 = rmsnorm(x, w[p + "post_attention_layernorm.weight"], self.eps)
                    g = F.linear(h2, w[p + "mlp.gate_proj.weight"], None)
                    u = F.linear(h2, w[p + "mlp.up_proj.weight"], None)
                    mlp = F.linear(swiglu(g, u), w[p + "mlp.down_proj.weight"], None)
                    x = x + mlp
                h = rmsnorm(x.reshape(1, -1), w["model.norm.weight"], self.eps)
                logits = F.linear(h.reshape(-1), lm_head_weight(w))
                seq_logits.append(logits)

        if not seq_logits:
            return torch.empty(0, self.vocab_size(), device=self.device)
        return torch.stack(seq_logits, dim=0)


def create_runner(config: EngineConfig, device: str | torch.device = "cpu", kind: str = "auto") -> ModelRunner:
    """Factory — real Qwen only.

    kind: 'qwen' | 'auto' (both return QwenRunner, fail loud on missing/OOM).
    'dummy' kind removed.
    """
    if kind == "dummy":
        raise ValueError("DummyRunner removed — use kind='qwen' or 'auto' with real Qwen weights")
    return QwenRunner(config, device=device)
