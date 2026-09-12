"""Simple block allocator for KV-cache pages (pure CPU bookkeeping).

Blocks are fixed-size token slots. Sequences hold one or more blocks;
alloc/free only tracks integer block ids so the same code works for the
Whisper and Qwen decode caches without touching torch.
"""
import threading


class BlockAllocator:
    """Fixed pool of equal-size blocks with alloc/free."""

    def __init__(self, num_blocks, block_size=16):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self._lock = threading.Lock()
        self._free = list(range(self.num_blocks - 1, -1, -1))
        self._owner = {}  # block_id -> owner tag

    def alloc(self, n, owner=""):
        """Allocate n block ids; raises MemoryError when the pool is short."""
        n = int(n)
        if n <= 0:
            return []
        with self._lock:
            if len(self._free) < n:
                raise MemoryError(
                    f"BlockAllocator: need {n} blocks, only {len(self._free)} free "
                    f"of {self.num_blocks} (block_size={self.block_size})"
                )
            ids = [self._free.pop() for _ in range(n)]
            for b in ids:
                self._owner[b] = owner
            return ids

    def free(self, ids):
        """Return block ids to the pool; unknown ids are ignored."""
        with self._lock:
            for b in ids:
                if b in self._owner:
                    del self._owner[b]
                    self._free.append(int(b))

    @property
    def free_count(self):
        """Number of currently free blocks."""
        with self._lock:
            return len(self._free)

    @property
    def used_count(self):
        """Number of currently allocated blocks."""
        with self._lock:
            return len(self._owner)

    def stats(self):
        """Dict with pool sizes and utilization."""
        with self._lock:
            used = len(self._owner)
            free = len(self._free)
        total = self.num_blocks
        return {
            "num_blocks": total,
            "block_size": self.block_size,
            "used": used,
            "free": free,
            "used_frac": (used / total) if total else 0.0,
        }

    def __len__(self):
        return self.num_blocks


class KVCacheAllocator:
    """Token-counting wrapper over BlockAllocator for one decode cache."""

    def __init__(self, num_blocks, block_size=16, name="kv"):
        self.name = name
        self.blocks = BlockAllocator(num_blocks, block_size)
        self._lock = threading.Lock()
        self._seqs = {}  # seq_id -> list[block_id]

    def _blocks_for(self, n_tokens):
        bs = self.blocks.block_size
        return (int(n_tokens) + bs - 1) // bs

    def alloc_seq(self, seq_id, n_tokens):
        """Allocate (or grow) the blocks backing seq_id for n_tokens."""
        need = self._blocks_for(n_tokens)
        with self._lock:
            have = self._seqs.get(seq_id, [])
            if len(have) >= need:
                return list(have)
        extra = self.blocks.alloc(need - len(have), owner=f"{self.name}:{seq_id}")
        with self._lock:
            self._seqs[seq_id] = list(self._seqs.get(seq_id, [])) + extra
            return list(self._seqs[seq_id])

    def free_seq(self, seq_id):
        """Release every block held by seq_id (no-op if unknown)."""
        with self._lock:
            ids = self._seqs.pop(seq_id, [])
        self.blocks.free(ids)

    def seq_blocks(self, seq_id):
        """Block ids currently backing seq_id (empty list if unknown)."""
        with self._lock:
            return list(self._seqs.get(seq_id, []))

    def stats(self):
        """Pool stats plus tracked-sequence count."""
        s = self.blocks.stats()
        with self._lock:
            s["sequences"] = len(self._seqs)
        s["name"] = self.name
        return s
