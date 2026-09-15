"""Engine configuration — no YAML, frozen dataclasses only."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "KVCacheConfig",
    "SchedulerConfig",
    "EngineConfig",
    "SamplingParams",
]


@dataclass(frozen=True)
class KVCacheConfig:
    """Paged KV-cache geometry."""

    block_size: int = 16  # tokens per block
    num_blocks: int = 512  # physical blocks
    num_layers: int = 28
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "fp16"  # fp16/bf16/fp32
    # derived: per-block bytes = block_size * num_kv_heads * head_dim * 2(K+V) * dtype_bytes

    def per_block_tokens(self) -> int:
        return int(self.block_size)

    def total_tokens(self) -> int:
        return int(self.num_blocks * self.block_size)


@dataclass(frozen=True)
class SchedulerConfig:
    """Continuous-batching scheduler limits."""

    max_num_seqs: int = 32  # max sequences in running+waiting
    max_num_batched_tokens: int = 512  # token budget per batch (prefill chunk)
    max_batch_size: int = 8  # max sequences per forward
    enable_chunked_prefill: bool = False  # full prefill by default; set True for chunked
    watermark_blocks: int = 2  # reserve blocks to avoid OOM


@dataclass(frozen=True)
class EngineConfig:
    """Top-level inference engine config."""

    model: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda"
    dtype: str = "fp16"
    max_seq_len: int = 512
    block_size: int = 16
    num_blocks: int = 256  # auto-derived from VRAM if 0
    max_batch_size: int = 8
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 512
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = True
    enable_cuda_graph: bool = True
    enable_fused_attention: bool = True
    # generation defaults
    default_max_tokens: int = 48
    # memory
    vram_budget_mb: float = 3800.0
    kv_cache_dtype: str = "fp16"  # fp16 | fp8

    def kv_config(self, num_layers=28, kv_heads=8, head_dim=128) -> KVCacheConfig:
        return KVCacheConfig(
            block_size=self.block_size,
            num_blocks=self.num_blocks or 256,
            num_layers=num_layers,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=self.kv_cache_dtype,
        )

    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_batch_size=self.max_batch_size,
            enable_chunked_prefill=self.enable_chunked_prefill,
        )


@dataclass(frozen=True)
class SamplingParams:
    """Per-request generation controls."""

    max_tokens: int = 48
    temperature: float = 0.0  # 0 => greedy
    top_p: float = 1.0
    top_k: int = -1  # -1 disabled
    stop_token_ids: tuple[int, ...] = field(default_factory=lambda: (151645, 151643))
    ignore_eos: bool = False

    def __post_init__(self):
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >=1")
