"""BlockManager — physical block allocation & memory tracking.

Mirrors vLLM's block manager but CPU-runnable. Manages a pool of
physical KV blocks; sequences map logical blocks -> physical via block_table.
Handles ref-counting for copy-on-write (fork) and OOM-aware admission.

Memory mgmt pillar: this is the allocator the scheduler consults before
admitting a sequence, and the component that recycles blocks on completion.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import torch

from src.inference.config import KVCacheConfig
from src.models.runtime.device import select_device
from src.models.runtime.tensor import parse_dtype

__all__ = ["BlockManager", "BlockSpaceManager"]


class BlockManager:
    """Manages physical KV-cache blocks."""

    def __init__(self, config: KVCacheConfig, enable_prefix_caching: bool = True):
        self.config = config
        self.num_blocks: int = int(config.num_blocks)
        self.block_size: int = int(config.block_size)
        # free stack (LIFO for cache locality)
        self._free: collections.deque[int] = collections.deque(range(self.num_blocks))
        # ref count per physical block (for CoW)
        self._ref: list[int] = [0] * self.num_blocks
        # block_table per seq_id -> list[physical]
        self._tables: dict[int, list[int]] = {}
        # prefix cache: hash(block_tokens) -> phys block id
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self._prefix_cache: dict[tuple[int, ...], int] = {}
        self._prefix_hits = 0
        self._prefix_misses = 0

    # -- introspection ------------------------------------------------
    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def used_count(self) -> int:
        return self.num_blocks - len(self._free)

    def available_blocks(self) -> int:
        return len(self._free)

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self._free) >= num_blocks

    # -- allocation ---------------------------------------------------
    def allocate(self, seq_id: int, num_blocks: int) -> list[int]:
        """Allocate `num_blocks` for seq_id, extending its table."""
        if num_blocks == 0:
            return self._tables.get(seq_id, [])
        if not self.can_allocate(num_blocks):
            raise RuntimeError(
                f"BlockManager OOM: need {num_blocks} blocks, "
                f"free={len(self._free)}/{self.num_blocks}"
            )
        table = self._tables.setdefault(seq_id, [])
        new_ids = [self._free.popleft() for _ in range(num_blocks)]
        for bid in new_ids:
            self._ref[bid] = 1
        table.extend(new_ids)
        return list(table)

    def ensure_for_tokens(self, seq_id: int, num_tokens: int) -> list[int]:
        """Ensure seq has enough blocks for `num_tokens` total."""
        needed = (num_tokens + self.block_size - 1) // self.block_size
        cur = len(self._tables.get(seq_id, []))
        if needed > cur:
            self.allocate(seq_id, needed - cur)
        return self._tables[seq_id]

    def append_token_maybe_new_block(self, seq_id: int, num_tokens_after: int) -> bool:
        """Appends one token; returns True if a new block was allocated."""
        # if the new token starts a new block, allocate one
        if (num_tokens_after - 1) % self.block_size == 0 and num_tokens_after > 1:
            # Actually first token of new block except very first block
            # Simpler: check needed vs current
            needed = (num_tokens_after + self.block_size - 1) // self.block_size
            cur = len(self._tables.get(seq_id, []))
            if needed > cur:
                self.allocate(seq_id, 1)
                return True
        # also handle first allocation lazily
        if seq_id not in self._tables:
            self.allocate(seq_id, 1)
            return True
        return False

    def ensure_seq(self, seq_id: int, seq_len: int):
        """Ensure blocks for seq_len tokens (prefill path)."""
        needed = (seq_len + self.block_size - 1) // self.block_size
        cur = len(self._tables.get(seq_id, []))
        if needed > cur:
            self.allocate(seq_id, needed - cur)

    # -- prefix caching -------------------------------------------------
    def try_reuse_prefix(self, seq_id: int, block_tokens: tuple[int, ...]) -> int | None:
        """Try to reuse a cached prefix block. Returns phys id or None."""
        if not self.enable_prefix_caching:
            return None
        h = tuple(block_tokens)
        bid = self._prefix_cache.get(h)
        if bid is not None and bid in self._free:
            # block was freed, not reusable
            self._prefix_cache.pop(h, None)
            return None
        if bid is not None:
            # reuse: bump ref, attach to seq
            self._ref[bid] += 1
            tbl = self._tables.setdefault(seq_id, [])
            tbl.append(bid)
            self._prefix_hits += 1
            return bid
        self._prefix_misses += 1
        return None

    def cache_prefix_block(self, block_tokens: tuple[int, ...], phys_id: int):
        if not self.enable_prefix_caching:
            return
        self._prefix_cache[tuple(block_tokens)] = phys_id

    def prefix_stats(self) -> dict:
        total = self._prefix_hits + self._prefix_misses
        return {"hits": self._prefix_hits, "misses": self._prefix_misses, "hit_rate": self._prefix_hits / max(total, 1), "cached_blocks": len(self._prefix_cache)}

    # -- fork / CoW ---------------------------------------------------
    def fork(self, parent_id: int, child_id: int):
        """Copy-on-write fork: child shares parent blocks initially."""
        parent_table = self._tables.get(parent_id)
        if parent_table is None:
            raise KeyError(f"parent seq {parent_id} has no blocks")
        self._tables[child_id] = list(parent_table)
        for bid in parent_table:
            self._ref[bid] += 1

    def _cow_for_write(self, seq_id: int, logical_idx: int):
        """If block is shared, copy it before write."""
        table = self._tables[seq_id]
        phys = table[logical_idx]
        if self._ref[phys] > 1:
            # allocate new physical block
            if not self._free:
                raise RuntimeError("CoW OOM: no free blocks")
            new_bid = self._free.popleft()
            self._ref[phys] -= 1
            self._ref[new_bid] = 1
            table[logical_idx] = new_bid
            return new_bid
        return phys

    def cow_if_needed(self, seq_id: int, pos: int):
        """Ensure the block containing pos is writable (CoW)."""
        logical = pos // self.block_size
        if logical < len(self._tables.get(seq_id, [])):
            self._cow_for_write(seq_id, logical)

    # -- freeing ------------------------------------------------------
    def free(self, seq_id: int):
        """Free all blocks for seq_id."""
        table = self._tables.pop(seq_id, [])
        for bid in table:
            self._ref[bid] -= 1
            if self._ref[bid] == 0:
                self._free.append(bid)
            elif self._ref[bid] < 0:
                raise RuntimeError(f"ref underflow block {bid}")

    def free_group(self, seq_ids: list[int]):
        for sid in seq_ids:
            self.free(sid)

    # -- stats --------------------------------------------------------
    def stats(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "free_blocks": len(self._free),
            "used_blocks": self.num_blocks - len(self._free),
            "block_size": self.block_size,
        }


# Alias for compatibility
BlockSpaceManager = BlockManager


@dataclass
class KVCacheMemoryPool:
    """Actual GPU tensor pool for paged KV.

    One pool per layer pair? We allocate [num_blocks, block_size, kv_heads, head_dim]
    per layer for K and V. On CPU this is still allocated (fallback).
    This is the 'memory mgmt' complement to BlockManager's policy.

    Shape per layer: K:[num_blocks, block_size, kv_heads, head_dim]
                     V:[num_blocks, block_size, kv_heads, head_dim]
    """

    config: KVCacheConfig
    device: torch.device
    dtype: torch.dtype

    # per-layer lists after init
    k_pools: list[torch.Tensor] | None = None
    v_pools: list[torch.Tensor] | None = None

    def __init__(self, config: KVCacheConfig, device: str | torch.device = "cpu", dtype: str | torch.dtype | None = None):
        self.config = config
        if isinstance(device, str):
            device = select_device(device)
        # fallback to cpu if cuda not avail
        try:
            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
        except Exception:
            device = torch.device("cpu")
        self.device = device
        torch_dtype = parse_dtype(dtype or config.dtype) if isinstance(dtype or config.dtype, str) else (dtype or config.dtype)
        if isinstance(torch_dtype, str):
            torch_dtype = parse_dtype(torch_dtype)
        # fp8 on CPU not supported -> fallback to fp16 for storage but report as fp8
        self._requested_dtype = torch_dtype
        if torch_dtype == torch.float8_e4m3fn and device.type == "cpu":
            # torch float8 requires cuda, fallback to fp16 on cpu but keep logical dtype as fp8 for mem calc
            torch_dtype = torch.float16
        if isinstance(torch_dtype, str):
            torch_dtype = parse_dtype(torch_dtype)
        self.dtype = torch_dtype
        self.k_pools = []
        self.v_pools = []
        for _ in range(config.num_layers):
            try:
                k = torch.zeros(
                    (config.num_blocks, config.block_size, config.num_kv_heads, config.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
            except Exception:
                # fallback if dtype not supported on device
                k = torch.zeros(
                    (config.num_blocks, config.block_size, config.num_kv_heads, config.head_dim),
                    dtype=torch.float16,
                    device=self.device,
                )
                self.dtype = torch.float16
            v = torch.zeros_like(k)
            self.k_pools.append(k)
            self.v_pools.append(v)

    def memory_mb(self) -> float:
        # Use requested dtype for accounting (fp8=1B even if fallback to fp16 on CPU)
        dtype_for_mem = getattr(self, "_requested_dtype", self.dtype)
        try:
            if dtype_for_mem == torch.float8_e4m3fn or str(dtype_for_mem) == "torch.float8_e4m3fn":
                bpe = 1
            else:
                bpe = torch.tensor([], dtype=dtype_for_mem).element_size() if hasattr(dtype_for_mem, "is_floating_point") and dtype_for_mem.is_floating_point else 2
                # fallback if not floating
                if bpe == 0:
                    bpe = 2
        except Exception:
            bpe = 1 if str(getattr(self, "_requested_dtype", "")).lower().find("fp8") >= 0 else 2
        # if requested was fp8 but we fell back to fp16, still report as fp8 for capacity planning
        if str(getattr(self, "_requested_dtype", "")).lower() in ("fp8", "fp8e4m3") or dtype_for_mem == torch.float8_e4m3fn:
            bpe = 1
        else:
            try:
                bpe = torch.tensor([], dtype=self.dtype).element_size()
            except Exception:
                bpe = 2
        total = 2 * self.config.num_blocks * self.config.block_size * self.config.num_kv_heads * self.config.head_dim * self.config.num_layers * bpe
        return total / (1024 ** 2)

    def store(self, layer: int, block_id: int, slot_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Store K/V at block_id, slot_idx. k,v shape [kv_heads, head_dim]."""
        # k,v are [kv_heads, head_dim] or [head_dim]
        if k.dim() == 1:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        self.k_pools[layer][block_id, slot_idx] = k.to(self.dtype).to(self.device)
        self.v_pools[layer][block_id, slot_idx] = v.to(self.dtype).to(self.device)

    def store_block(self, layer: int, block_id: int, k_block: torch.Tensor, v_block: torch.Tensor):
        """Store whole block: [block_size, kv_heads, head_dim]"""
        n = k_block.shape[0]
        self.k_pools[layer][block_id, :n] = k_block.to(self.dtype).to(self.device)
        self.v_pools[layer][block_id, :n] = v_block.to(self.dtype).to(self.device)

    def gather(self, layer: int, block_table: list[int], seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for a sequence -> [seq_len, kv_heads, head_dim] contiguous."""
        if seq_len == 0:
            # empty
            return (
                torch.empty(0, self.config.num_kv_heads, self.config.head_dim, device=self.device, dtype=self.dtype),
                torch.empty(0, self.config.num_kv_heads, self.config.head_dim, device=self.device, dtype=self.dtype),
            )
        bs = self.config.block_size
        parts_k = []
        parts_v = []
        remaining = seq_len
        for bid in block_table:
            take = min(bs, remaining)
            parts_k.append(self.k_pools[layer][bid, :take])
            parts_v.append(self.v_pools[layer][bid, :take])
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            raise RuntimeError(f"gather: block_table too short for seq_len {seq_len}")
        return torch.cat(parts_k, dim=0), torch.cat(parts_v, dim=0)

    def clear_block(self, block_id: int):
        for k, v in zip(self.k_pools, self.v_pools):
            k[block_id].zero_()
            v[block_id].zero_()
