"""Tensor helpers: dtype parsing, device moves, cache control."""
import numpy as np
import torch

DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp8e4m3": torch.float8_e4m3fn,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}

_DTYPE_TO_NAME = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.int8: "int8",
    torch.int32: "int32",
    torch.int64: "int64",
}


def parse_dtype(name):
    """Resolve a dtype name or torch.dtype to a torch.dtype."""
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower().replace("torch.", "")
    if key not in DTYPE_MAP:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[key]


def dtype_name(dtype):
    """Short canonical name for a torch.dtype."""
    return _DTYPE_TO_NAME.get(dtype, str(dtype))


def to_device(tensor, device):
    """Move a tensor to device (no-op if already there)."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"to_device expects a Tensor, got {type(tensor)}")
    target = torch.device(device)
    if tensor.device == target:
        return tensor
    return tensor.to(target)


def to_dtype(tensor, dtype):
    """Cast a tensor to dtype (no-op if already that dtype)."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"to_dtype expects a Tensor, got {type(tensor)}")
    dtype = parse_dtype(dtype)
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


def move(tensor, device=None, dtype=None, non_blocking=False):
    """Move and/or cast a tensor in one call."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"move expects a Tensor, got {type(tensor)}")
    if dtype is not None:
        dtype = parse_dtype(dtype)
    if device is None and dtype is None:
        return tensor
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)


def move_dict(batch, device=None, dtype=None, non_blocking=False):
    """Move every Tensor value in a dict; pass through non-tensors."""
    return {
        k: (move(v, device, dtype, non_blocking) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def empty_cache():
    """Release cached CUDA blocks back to the driver (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_host_numpy(x, dtype=np.float32):
    """Anything array-like (numpy, list, CPU/CUDA tensor) -> host np array.

    The single choke point for host boundaries (mel frontend, VAD, wav io)
    so callers never hit 'can't convert cuda tensor to numpy'.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu()
    return np.asarray(x, dtype=dtype)
