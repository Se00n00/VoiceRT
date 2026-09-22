"""Minimal GGUF reader for MiniCPM5-1B Q4_K_M (no new deps).

Parses the GGUFv3 container: header, metadata KV, tensor table with
per-tensor quant types and file offsets. Returns raw row bytes; the
Q4_K/Q6_K Triton kernels in
:mod:`src.models.triton_kernels.minicpm_q4k` consume them directly —
weights stay packed from disk to GPU.
"""
import struct

__all__ = ["GGML_TYPES", "BLOCK_BYTES", "read_gguf", "tensor_inventory"]

MAGIC = b"GGUF"

# ggml_type -> (name, block_elems, block_bytes or None if dense)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q8_0", 32, 34),
    7: ("Q5_0", 32, 22),
    8: ("Q5_1", 32, 24),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    24: ("IQ4_XS", 256, 120),
    30: ("BF16", 1, 2),
}

BLOCK_BYTES = {k: v[2] for k, v in GGML_TYPES.items()}


def _u32(f):
    return struct.unpack("<I", f.read(4))[0]


def _u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def _str(f):
    n = _u64(f)
    return f.read(n).decode("utf-8", errors="replace")


def _skip_value(f, vtype):
    # GGML metadata value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32,
    # 6 f32, 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64.
    if vtype in (0, 1, 7):
        f.read(1)
    elif vtype in (2, 3):
        f.read(2)
    elif vtype in (4, 5, 6):
        f.read(4)
    elif vtype in (10, 11, 12):
        f.read(8)
    elif vtype == 8:  # string
        _str(f)
    elif vtype == 9:  # array
        etype = _u32(f)
        n = _u64(f)
        for _ in range(n):
            _skip_value(f, etype)
    else:
        raise ValueError(f"unknown gguf metadata type {vtype}")


def _read_metadata(f, n_kv):
    meta = {}
    for _ in range(n_kv):
        key = _str(f)
        vtype = _u32(f)
        if vtype == 8:
            meta[key] = _str(f)
        elif vtype == 4:
            meta[key] = struct.unpack("<i", f.read(4))[0]
        elif vtype == 6:
            meta[key] = struct.unpack("<f", f.read(4))[0]
        else:
            _skip_value(f, vtype)
            meta[key] = None
    return meta


def read_gguf(path):
    """Parse GGUF file -> {"meta": dict, "tensors": [{name, dtype, shape,
    offset, nbytes}], "data_start": int} (offsets absolute file bytes)."""
    tensors = []
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("not a GGUF file")
        version = _u32(f)
        if version != 3:
            raise ValueError(f"supports GGUFv3, got v{version}")
        n_tensors = _u64(f)
        n_kv = _u64(f)
        meta = _read_metadata(f, n_kv)
        infos = []
        for _ in range(n_tensors):
            name = _str(f)
            n_dims = _u32(f)
            dims = [_u64(f) for _ in range(n_dims)]
            dtype = _u32(f)
            offset = _u64(f)
            infos.append((name, dtype, dims, offset))
        # data section starts 32-byte aligned after the header
        data_start = (f.tell() + 31) // 32 * 32
        for name, dtype, dims, offset in infos:
            n_elem = 1
            for d in dims:
                n_elem *= d
            if dtype in GGML_TYPES:
                _, block_elems, block_bytes = GGML_TYPES[dtype]
                nbytes = n_elem // block_elems * block_bytes
            else:
                raise ValueError(f"tensor {name}: unknown dtype {dtype}")
            tensors.append({"name": name, "dtype": dtype,
                            "type": GGML_TYPES[dtype][0],
                            "shape": tuple(dims), "offset": data_start + offset,
                            "nbytes": nbytes, "nelem": n_elem})
    return {"meta": meta, "tensors": tensors, "data_start": data_start,
            "path": path}


def tensor_inventory(parsed):
    """Human-readable per-type summary + full name list."""
    from collections import Counter

    counts: Counter = Counter()
    nbytes = 0
    for t in parsed["tensors"]:
        counts[t["type"]] += 1
        nbytes += t["nbytes"]
    return {"n_tensors": len(parsed["tensors"]),
            "by_type": dict(counts),
            "total_mb": nbytes / (1024 ** 2),
            "names": [t["name"] for t in parsed["tensors"]]}
