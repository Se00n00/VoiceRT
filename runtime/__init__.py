"""runtime: device/memory/profiler/tensor helpers + capacity + scheduler.

Wired into serving: device + profiler + tensor.to_host_numpy are in the
engine/server hot path; memory.check_budget guards every voice turn and
scheduler.FIFOScheduler is the server admission queue; capacity plans
sessions from VRAM at startup. Everything exported here has a live caller
or a unit test — no decorative surface.
"""
from runtime.capacity import estimate_session_mb, kv_cache_mb, plan_capacity, probe_vram
from runtime.device import (
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
from runtime.memory import MemoryBudgetExceeded, check_budget, fits, summary
from runtime.profiler import Profiler
from runtime.scheduler import FIFOScheduler
from runtime.tensor import (
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
