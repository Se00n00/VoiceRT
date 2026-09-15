"""InferenceEngine package — scheduling, batching, KV cache, memory mgmt.

Public surface:

    from src.inference import InferenceEngine, EngineConfig, SamplingParams

    engine = InferenceEngine(EngineConfig(model="Qwen/Qwen3-0.6B"))
    engine.add_request("r1", prompt_ids, SamplingParams(max_tokens=32))
    while engine.has_unfinished():
        outs = engine.step()

Diagram:

    ┌───────▼───────┐
    │ Inference     │
    │ Engine        │
    │  scheduling   │  <- src.inference.scheduler.ContinuousScheduler
    │  batching     │  <- src.inference.batching.make_batch
    │  KV cache     │  <- src.inference.kv_cache.PagedKVCache
    │  memory mgmt  │  <- src.inference.block_manager.BlockManager
    └───────┬───────┘
"""

from src.inference.config import EngineConfig, KVCacheConfig, SamplingParams, SchedulerConfig
from src.inference.engine import InferenceEngine
from src.inference.block_manager import BlockManager, KVCacheMemoryPool
from src.inference.kv_cache import PagedKVCache
from src.inference.scheduler import ContinuousScheduler
from src.inference.sequence import Sequence, SequenceGroup, SequenceStatus
from src.inference.model_runner import QwenRunner

__all__ = [
    "EngineConfig",
    "KVCacheConfig",
    "SchedulerConfig",
    "SamplingParams",
    "InferenceEngine",
    "BlockManager",
    "KVCacheMemoryPool",
    "PagedKVCache",
    "ContinuousScheduler",
    "Sequence",
    "SequenceGroup",
    "SequenceStatus",
    "QwenRunner",
]
