"""Paged KV-cache facade — thin wrapper over BlockManager + MemoryPool.

This module is the 'KV cache' box in the diagram. It owns the paged
storage and the gather/store primitives the attention kernels call.
"""

from __future__ import annotations

import torch

from src.inference.block_manager import BlockManager, KVCacheMemoryPool
from src.inference.config import KVCacheConfig

__all__ = ["PagedKVCache"]


class PagedKVCache:
    """Paged KV-cache with block-table indirection.

    Real tensors, real block tables, real scatter/gather. Attention never
    sees a flat contiguous KV — it must gather via this cache, exercising
    the paged path even on CPU.
    """

    def __init__(self, config: KVCacheConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.block_manager = BlockManager(config)
        self.pool = KVCacheMemoryPool(config, device=device, dtype=config.dtype)
        self.device = self.pool.device
        self.dtype = self.pool.dtype

    # -- allocation helpers -------------------------------------------
    def allocate_for_seq(self, seq_id: int, seq_len: int):
        """Ensure blocks for seq_len tokens."""
        self.block_manager.ensure_seq(seq_id, seq_len)

    def append_slot(self, seq_id: int, new_seq_len: int):
        """Called after appending one token; allocates new block if needed."""
        needed = (new_seq_len + self.config.block_size - 1) // self.config.block_size
        cur = len(self.block_manager._tables.get(seq_id, []))
        if needed > cur:
            self.block_manager.allocate(seq_id, 1)

    def free(self, seq_id: int):
        self.block_manager.free(seq_id)
        # optional: clear tensors for hygiene

    @property
    def free_blocks(self) -> int:
        return self.block_manager.free_count

    @property
    def used_blocks(self) -> int:
        return self.block_manager.used_count

    def can_allocate_seq(self, seq_len: int) -> bool:
        need = (seq_len + self.config.block_size - 1) // self.config.block_size
        return self.block_manager.can_allocate(need)

    # -- low-level K/V ops --------------------------------------------
    def store_prefill(self, seq_id: int, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Store full prefill K/V for a sequence.

        k,v: [seq_len, kv_heads, head_dim]
        Scatter into paged blocks according to block_table.
        """
        seq_len = k.shape[0]
        table = self.block_manager._tables[seq_id]
        bs = self.config.block_size
        off = 0
        for bid in table:
            take = min(bs, seq_len - off)
            if take <= 0:
                break
            self.pool.store_block(layer, bid, k[off : off + take], v[off : off + take])
            off += take

    def store_decode(self, seq_id: int, layer: int, pos: int, k1: torch.Tensor, v1: torch.Tensor):
        """Store single decode token K/V at absolute position pos.

        k1,v1: [kv_heads, head_dim]
        """
        # CoW if needed
        self.block_manager.cow_if_needed(seq_id, pos)
        table = self.block_manager._tables[seq_id]
        bs = self.config.block_size
        bid = table[pos // bs]
        slot = pos % bs
        self.pool.store(layer, bid, slot, k1, v1)

    def gather(self, seq_id: int, layer: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather contiguous K/V for seq up to seq_len."""
        table = self.block_manager._tables[seq_id]
        return self.pool.gather(layer, table, seq_len)

    def stats(self) -> dict:
        return {
            **self.block_manager.stats(),
            "memory_mb": self.pool.memory_mb(),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
