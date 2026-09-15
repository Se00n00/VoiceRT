"""Runtime facade for the inference engine.

This module is the integration point between the existing runtime
(device / memory / capacity / scheduler) and the new paged inference engine.

It re-exports the real engine and adds helpers that connect VRAM probing
to KV block sizing — the 'memory mgmt' pillar that plans capacity at boot
like the existing capacity.py planner.

All logic here lives on top of src.inference; nothing is simulated.
"""

from __future__ import annotations

from src.inference import (
    BlockManager,
    EngineConfig,
    InferenceEngine,
    KVCacheConfig,
    PagedKVCache,
    SamplingParams,
    SchedulerConfig,
)
from src.inference.block_manager import KVCacheMemoryPool
from src.inference.engine import EngineOutput
from src.models.runtime.capacity import estimate_session_mb, kv_cache_mb, plan_capacity, probe_vram
from src.models.runtime.device import allocated_mb, max_allocated_mb, reserved_mb, select_device
from src.models.runtime.memory import MemoryBudgetExceeded, check_budget, fits
from src.models.runtime.scheduler import FIFOScheduler

__all__ = [
    "InferenceEngine",
    "EngineConfig",
    "KVCacheConfig",
    "SchedulerConfig",
    "SamplingParams",
    "EngineOutput",
    "PagedKVCache",
    "BlockManager",
    "KVCacheMemoryPool",
    "FIFOScheduler",
    "probe_vram",
    "plan_capacity",
    "estimate_session_mb",
    "kv_cache_mb",
    "select_device",
    "allocated_mb",
    "max_allocated_mb",
    "reserved_mb",
    "check_budget",
    "fits",
    "MemoryBudgetExceeded",
]


def auto_engine_config(
    model: str = "Qwen/Qwen3-0.6B",
    max_tokens: int = 48,
    vram_budget_mb: float | None = None,
) -> EngineConfig:
    """Derive EngineConfig from VRAM probe + capacity planner.

    Mirrors VoiceAgent boot: probe VRAM, price a session, set num_blocks
    and max_batch_size from the serving plan. Never raises.
    """
    try:
        info = probe_vram()
        total = float(info.get("total_mb") or 0)
        # estimate baseline as if weights already resident (heuristic)
        # we don't have baseline yet, assume 1800 MB for Qwen+weights
        baseline = 1800.0
        plan = plan_capacity(total, baseline, max_new_tokens=max_tokens)
        # num_blocks ~ sessions * tokens_per_session / block_size
        sessions = int(plan.get("max_sessions", 1))
        block_size = 16
        tokens_per_session = max_tokens + 200  # prompt + gen
        blocks_per_session = (tokens_per_session + block_size - 1) // block_size
        num_blocks = max(32, sessions * blocks_per_session * 2)  # 2x for safety
        num_blocks = min(num_blocks, 1024)
        budget = vram_budget_mb or float(plan.get("vram_total_mb") or 3800)
        return EngineConfig(
            model=model,
            num_blocks=num_blocks,
            block_size=block_size,
            max_batch_size=min(8, max(1, sessions)),
            default_max_tokens=max_tokens,
            vram_budget_mb=budget,
        )
    except Exception:
        return EngineConfig(model=model, default_max_tokens=max_tokens)
