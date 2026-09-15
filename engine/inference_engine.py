"""engine/inference_engine — top-level re-export for the diagram.

The diagram's 'Inference Engine' box with scheduling / batching / KV cache /
memory mgmt is implemented in src.inference and re-exported here so that
both the voice-pipeline 'engine/' package and 'src/' package resolve it.

Import either way:

    from engine.inference_engine import InferenceEngine, EngineConfig
    from src.inference import InferenceEngine
    from src.models.runtime.inference_engine import InferenceEngine

All are the same real engine (torch, paged KV, continuous batching).
"""

from src.inference import (
    BlockManager,
    EngineConfig,
    InferenceEngine,
    KVCacheConfig,
    PagedKVCache,
    SamplingParams,
    SchedulerConfig,
)
from src.inference.engine import EngineOutput

__all__ = [
    "InferenceEngine",
    "EngineConfig",
    "KVCacheConfig",
    "SchedulerConfig",
    "SamplingParams",
    "EngineOutput",
    "BlockManager",
    "PagedKVCache",
]
