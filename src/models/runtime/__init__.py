"""runtime: device/memory/profiler/tensor helpers + capacity + scheduler.

Wired into serving: device + profiler + tensor.to_host_numpy are in the
engine/server hot path; memory.check_budget guards every voice turn and
scheduler.FIFOScheduler is the server admission queue; capacity plans
sessions from VRAM at startup. Everything exported here has a live caller
or a unit test — no decorative surface.
"""
from src.models.runtime.capacity import estimate_session_mb, kv_cache_mb, plan_capacity, probe_vram
from src.models.runtime.device import (
    allocated_mb,
    current_device,
    is_cuda_available,
    max_allocated_mb,
    reserved_mb,
    reset_peak_stats,
    select_device,
    synchronize,
    vram_stats,
)
from src.models.runtime.memory import MemoryBudgetExceeded, check_budget, fits, summary
from src.models.runtime.profiler import Profiler
from src.models.runtime.scheduler import FIFOScheduler

# inference engine — available via explicit import to avoid circular deps:
#   from src.inference import InferenceEngine
#   from src.models.runtime.inference_engine import InferenceEngine, auto_engine_config
# (runtime/__init__ does not eagerly import inference to keep import graph acyclic;
#  the facade module `runtime/inference_engine.py` re-exports with capacity-aware helpers.)
from src.models.runtime.tensor import (
    DTYPE_MAP,
    dtype_name,
    empty_cache,
    move,
    move_dict,
    parse_dtype,
    to_device,
    to_dtype,
    to_host_numpy,
)

__all__ = [
    "estimate_session_mb",
    "kv_cache_mb",
    "plan_capacity",
    "probe_vram",
    "allocated_mb",
    "current_device",
    "is_cuda_available",
    "max_allocated_mb",
    "reserved_mb",
    "reset_peak_stats",
    "select_device",
    "synchronize",
    "vram_stats",
    "MemoryBudgetExceeded",
    "check_budget",
    "fits",
    "summary",
    "Profiler",
    "FIFOScheduler",
    "DTYPE_MAP",
    "dtype_name",
    "empty_cache",
    "move",
    "move_dict",
    "parse_dtype",
    "to_device",
    "to_dtype",
    "to_host_numpy",
]
