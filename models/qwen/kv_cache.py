"""KV-cache for Qwen greedy decoding: preallocated per-layer K/V buffers."""
import torch

__all__ = ["KVCache"]


class KVCache:
    """Per-layer key/value cache, shape [nlayers, Hk, max_len, dh] each."""

    def __init__(self, nlayers, n_kv_heads, max_len, head_dim,
                 device="cuda:0", dtype=None):
        self.nlayers = int(nlayers)
        self.n_kv_heads = int(n_kv_heads)
        self.max_len = int(max_len)
        self.head_dim = int(head_dim)
        self.device = device
        dev = torch.device(device)
        kw = {"device": dev}
        if dtype is not None:
            kw["dtype"] = dtype
        self.K = [torch.empty(n_kv_heads, max_len, head_dim, **kw)
                  for _ in range(nlayers)]
        self.V = [torch.empty(n_kv_heads, max_len, head_dim, **kw)
                  for _ in range(nlayers)]
        self.lengths = [0] * nlayers

    def store_prefill(self, layer, k, v):
        """Store prefill K/V [T, Hk, dh] at positions 0..T-1."""
        T = k.shape[0]
        self.K[layer][:, :T] = k.transpose(0, 1)
        self.V[layer][:, :T] = v.transpose(0, 1)
        self.lengths[layer] = T

    def store_step(self, layer, pos, k, v):
        """Store one decode step K/V [Hk, dh] at ``pos``."""
        self.K[layer][:, pos] = k
        self.V[layer][:, pos] = v
        self.lengths[layer] = max(self.lengths[layer], pos + 1)

    def get(self, layer, end=None):
        """(K, V) slices [Hk, N, dh] for ``N = end`` (default: filled)."""
        n = self.lengths[layer] if end is None else int(end)
        return self.K[layer][:, :n], self.V[layer][:, :n]

    def reset(self):
        self.lengths = [0] * self.nlayers
