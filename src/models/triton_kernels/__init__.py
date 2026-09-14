"""Ported hand-written Triton kernels (absolute-import package).

Low-level kernels (one file per op) plus the per-leg surfaces
(src.models.triton_kernels.qwen / .whisper / .tts) that the model legs import.
Everything exported here is imported by a model leg, a test, or both.
"""
from src.models.triton_kernels.utils import grid1d, next_pow2, cdiv, num_warps_for_width, tl_dtype, check_cuda
from src.models.triton_kernels.activation import swiglu, silu, _swiglu_kernel, _silu_kernel
from src.models.triton_kernels.activation import lstm_cell, _lstm_cell_kernel
from src.models.triton_kernels.attention import decode_attn, batched_decode_attn, gqa_decode_attn
from src.models.triton_kernels.attention import _dec_attn_kernel, _bdec_attn_kernel, _gqa_dec_kernel
from src.models.triton_kernels.attention import fused_qkv, _fused_qkv_kernel, fused_qkv_gqa, _fused_qkv_gqa_kernel
from src.models.triton_kernels.layernorm import layernorm, _ln_kernel
from src.models.triton_kernels.rmsnorm import rmsnorm, _rmsnorm_kernel
from src.models.triton_kernels.softmax import row_softmax, _softmax_kernel
from src.models.triton_kernels.rope import rope, rope_batched, _rope_kernel, _rope_batch_kernel
from src.models.triton_kernels.conv1d import conv1d_silu, conv1d_silu_fwd_kernel
from src.models.triton_kernels.conv1d import in1d_silu, _in1d_silu_kernel

__all__ = [
    "grid1d", "next_pow2", "cdiv", "num_warps_for_width", "tl_dtype", "check_cuda",
    "swiglu", "silu", "_swiglu_kernel", "_silu_kernel",
    "lstm_cell", "_lstm_cell_kernel",
    "decode_attn", "batched_decode_attn", "gqa_decode_attn",
    "_dec_attn_kernel", "_bdec_attn_kernel", "_gqa_dec_kernel",
    "fused_qkv", "_fused_qkv_kernel", "fused_qkv_gqa", "_fused_qkv_gqa_kernel",
    "layernorm", "_ln_kernel",
    "rmsnorm", "_rmsnorm_kernel",
    "row_softmax", "_softmax_kernel",
    "rope", "rope_batched", "_rope_kernel", "_rope_batch_kernel",
    "conv1d_silu", "conv1d_silu_fwd_kernel",
    "in1d_silu", "_in1d_silu_kernel",
]
