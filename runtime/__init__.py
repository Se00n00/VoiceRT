"""runtime: device/memory/stream/scheduler/batcher/profiler helpers."""
from runtime.allocator import BlockAllocator, KVCacheAllocator
from runtime.batcher import MicroBatcher, batch_by_leg, group_by_leg, split_batches
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
from runtime.event import TimerEvent, elapsed_ms, now_event, time_block
from runtime.graph import CudaGraphRunner, graphs_supported
from runtime.memory import MemoryBudget, MemoryBudgetExceeded, check_budget, fits, summary
from runtime.profiler import Profiler
from runtime.request import ChatRequest, SpeakRequest, TranscribeRequest, TurnResult, VadRequest
from runtime.scheduler import FIFOScheduler
from runtime.stream import CudaStream, current_stream, synchronize_all
from runtime.tensor import (
    DTYPE_MAP,
    dtype_name,
    empty_cache,
    move,
    move_dict,
    parse_dtype,
    to_device,
    to_dtype,
)

__all__ = [
    "BlockAllocator",
    "KVCacheAllocator",
    "MicroBatcher",
    "batch_by_leg",
    "group_by_leg",
    "split_batches",
    "allocated_mb",
    "current_device",
    "is_cuda_available",
    "max_allocated_mb",
    "reserved_mb",
    "reset_peak_stats",
    "select_device",
    "synchronize",
    "vram_stats",
    "TimerEvent",
    "elapsed_ms",
    "now_event",
    "time_block",
    "CudaGraphRunner",
    "graphs_supported",
    "MemoryBudget",
    "MemoryBudgetExceeded",
    "check_budget",
    "fits",
    "summary",
    "Profiler",
    "ChatRequest",
    "SpeakRequest",
    "TranscribeRequest",
    "TurnResult",
    "VadRequest",
    "FIFOScheduler",
    "CudaStream",
    "current_stream",
    "synchronize_all",
    "DTYPE_MAP",
    "dtype_name",
    "empty_cache",
    "move",
    "move_dict",
    "parse_dtype",
    "to_device",
    "to_dtype",
]
