"""Hand-written Triton kernels for the Qwen3-0.6B LLM leg (absolute-import package).

Qwen-only: low-level ops (rmsnorm/rope/swiglu/attention) plus the per-leg
surfaces (`qwen`, `qwen_fused`) and `paged_attention` for the paged runner.
STT/TTS run pure torch (`src.models.pytorch.whisper/tts`) — no kernels here.
Everything exported is imported by the Qwen leg, the paged runner, or a test.
"""
from src.models.triton_kernels.utils import grid1d, next_pow2, cdiv, num_warps_for_width, tl_dtype, check_cuda
from src.models.triton_kernels.activation import swiglu, silu, _swiglu_kernel, _silu_kernel
from src.models.triton_kernels.attention import decode_attn, batched_decode_attn, gqa_decode_attn
from src.models.triton_kernels.attention import _dec_attn_kernel, _bdec_attn_kernel, _gqa_dec_kernel
from src.models.triton_kernels.attention import fused_qkv, _fused_qkv_kernel, fused_qkv_gqa, _fused_qkv_gqa_kernel
from src.models.triton_kernels.rmsnorm import rmsnorm, _rmsnorm_kernel
from src.models.triton_kernels.rope import rope, rope_batched, _rope_kernel, _rope_batch_kernel

__all__ = [
    "grid1d", "next_pow2", "cdiv", "num_warps_for_width", "tl_dtype", "check_cuda",
    "swiglu", "silu", "_swiglu_kernel", "_silu_kernel",
    "decode_attn", "batched_decode_attn", "gqa_decode_attn",
    "_dec_attn_kernel", "_bdec_attn_kernel", "_gqa_dec_kernel",
    "fused_qkv", "_fused_qkv_kernel", "fused_qkv_gqa", "_fused_qkv_gqa_kernel",
    "rmsnorm", "_rmsnorm_kernel",
    "rope", "rope_batched", "_rope_kernel", "_rope_batch_kernel",
]
