"""Prefix caching — block-level dedup for shared prefixes (system prompt + history).

Voice pipeline: 20turns * 248tok history + system prompt share prefix across session turns.
This module hashes token blocks and reuses physical blocks via BlockManager ref-counting.
"""
from __future__ import annotations
import hashlib
from typing import Dict, Tuple

class PrefixCache:
    """Hash -> physical block id for prefix blocks.

    Each block is `block_size` tokens (16). For a prefix of N blocks, we store
    mapping from block_hash to physical block id. On new request, we probe
    prefix blocks in order; hit -> fork (ref++), miss -> allocate new and insert.
    """
    def __init__(self, block_size: int = 16):
        self.block_size = int(block_size)
        self._table: Dict[Tuple[int, ...], int] = {}  # hash -> phys block
        self._hits = 0
        self._misses = 0

    def hash_block(self, tokens: Tuple[int, ...]) -> Tuple[int, ...]:
        # block hash as tuple of tokens (or could be xxhash)
        return tuple(tokens)

    def probe(self, block_tokens: Tuple[int, ...]) -> int | None:
        h = self.hash_block(block_tokens)
        bid = self._table.get(h)
        if bid is not None:
            self._hits += 1
            return bid
        self._misses += 1
        return None

    def insert(self, block_tokens: Tuple[int, ...], phys_id: int):
        h = self.hash_block(block_tokens)
        self._table[h] = phys_id

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses, "hit_rate": self._hits / max(total, 1), "size": len(self._table)}

    def clear(self):
        self._table.clear()
        self._hits = self._misses = 0
