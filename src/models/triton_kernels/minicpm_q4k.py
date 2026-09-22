"""Single-file Q4_K / Q6_K kernels for MiniCPM5-1B Q4_K_M (from scratch).

Covers the two quant types in a Q4_K_M mix: Q4_K for most projections,
Q6_K for precision-sensitive ones (down-projections). Layouts follow the
GGML k-quant spec, verified against independent bit-level references:

- Q4_K super-block (256 weights, 144 B, 4.5 bits/weight):
  ``d:f16 | dmin:f16 | scales:u8[12] | qs:u8[128]``.
  8 sub-blocks of 32 weights; sub-block j holds the low nibbles of
  chunk j//2 when j is even, the high nibbles when j is odd.
  6-bit scale/min per sub-block via ``get_scale_min_k4``:
  j<4 -> (scales[j]&63, scales[j+4]&63), else packed from bytes j+4/j-4/j.
  weight = d*sc*q - dmin*m.
- Q6_K super-block (256 weights, 210 B, 6.5625 bits/weight):
  ``ql:u8[128] | qh:u8[64] | scales:i8[16] | d:f16``.
  Two 128-weight halves; within a half, for l in 0..32 with is=l//16:
  q1..q4 combine ql nibbles with 2-bit qh slices, minus 32, times the
  int8 scale sc[is + {0,2,4,6}] and the block scale d.

Decode scope (this file): fused dequant-GEMV ``y = W x`` with W packed.
Prefill goes dequantize-then-dense via the torch reference below.
Host convention (keeps Triton free of fp16 bitcasts): per-row ``d`` (and
``dmin`` for Q4_K) are precomputed fp32 on the host; the kernel unpacks
only nibbles + scales/mins. KV cache and activations stay fp16/bf16.

All kernels in ONE file per spec. Includes @triton.testing.perf_report
inside file, a hand-computed exact test vector (V1 gate), and a random
parity test. Qwen path untouched; selected via ``LlmConfig.backend``.
"""
import math
import torch
import triton
import triton.language as tl

from src.models.runtime.memory import check_budget

HAVE_TRITON = True
try:
    import triton  # noqa
except Exception:
    HAVE_TRITON = False

QK_K = 256          # weights per k-quant super-block
Q4K_BYTES = 144     # 2 + 2 + 12 + 128
Q6K_BYTES = 210     # 128 + 64 + 16 + 2
SUB = 32            # weights per Q4_K sub-block


def _cuda(t):
    return isinstance(t, torch.Tensor) and t.is_cuda


# ---------- torch exact reference (oracle for parity + prefill path) ----------
def get_scale_min_k4(scales):
    """scales u8[..., 8, 12] -> (sc, m) u8[..., 8, 8] each, exact per spec."""
    sc = torch.empty(scales.shape[:-1] + (8,), dtype=torch.uint8, device=scales.device)
    m = torch.empty_like(sc)
    sc[..., :4] = scales[..., :4] & 63
    m[..., :4] = scales[..., 4:8] & 63
    for j in range(4, 8):
        sc[..., j] = (scales[..., j + 4] & 0x0F) | ((scales[..., j - 4] >> 6) << 4)
        m[..., j] = (scales[..., j + 4] >> 4) | ((scales[..., j] >> 6) << 4)
    return sc, m


def dequantize_q4_k_torch(blocks, out_dtype=torch.float16):
    """blocks [R, Nb*144] u8 -> [R, Nb*256] fp. Vectorized, exact."""
    R = blocks.shape[0]
    total = blocks.shape[1] // Q4K_BYTES
    t = blocks.reshape(R, total, Q4K_BYTES)
    d = t[..., 0:2].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)
    dmin = t[..., 2:4].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)
    s = t[..., 4:16]
    q = t[..., 16:144].reshape(R, total, 4, SUB)
    sc, m = get_scale_min_k4(s)
    out = torch.empty(R, total, 8, SUB, dtype=torch.float32, device=blocks.device)
    d = d[..., None]
    dmin = dmin[..., None]
    sc = sc.to(torch.float32)
    m = m.to(torch.float32)
    for c in range(4):
        low = (q[..., c, :] & 0x0F).to(torch.float32)
        high = ((q[..., c, :] >> 4) & 0x0F).to(torch.float32)
        out[..., 2 * c, :] = d * sc[..., 2 * c, None] * low \
            - dmin * m[..., 2 * c, None]
        out[..., 2 * c + 1, :] = d * sc[..., 2 * c + 1, None] * high \
            - dmin * m[..., 2 * c + 1, None]
    return out.reshape(R, total * QK_K).to(out_dtype)


def dequantize_q6_k_torch(blocks, out_dtype=torch.float16):
    """blocks [R, Nb*210] u8 -> [R, Nb*256] fp. Vectorized, exact."""
    R = blocks.shape[0]
    total = blocks.shape[1] // Q6K_BYTES
    t = blocks.reshape(R, total, Q6K_BYTES)
    ql = t[..., 0:128].reshape(R, total, 2, 64)
    qh = t[..., 128:192].reshape(R, total, 2, 32)
    sc = t[..., 192:208].reshape(R, total, 2, 8).to(torch.int8).to(torch.float32)
    d = t[..., 208:210].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)
    d = d[..., None]
    out = torch.empty(R, total, 2, 128, dtype=torch.float32, device=blocks.device)
    ar = torch.arange(32, device=blocks.device)
    is_ = (ar // 16).to(torch.int64)
    for n in range(2):
        q1 = ((ql[..., n, :32] & 0x0F) | (((qh[..., n, :] >> 0) & 3) << 4)).to(torch.float32) - 32.0
        q2 = ((ql[..., n, 32:] & 0x0F) | (((qh[..., n, :] >> 2) & 3) << 4)).to(torch.float32) - 32.0
        q3 = ((ql[..., n, :32] >> 4) | (((qh[..., n, :] >> 4) & 3) << 4)).to(torch.float32) - 32.0
        q4 = ((ql[..., n, 32:] >> 4) | (((qh[..., n, :] >> 6) & 3) << 4)).to(torch.float32) - 32.0
        base = d * sc[..., n, :][..., is_]
        out[..., n, 0:32] = base * q1
        out[..., n, 32:64] = d * sc[..., n, :][..., is_ + 2] * q2
        out[..., n, 64:96] = d * sc[..., n, :][..., is_ + 4] * q3
        out[..., n, 96:128] = d * sc[..., n, :][..., is_ + 6] * q4
    return out.reshape(R, total * QK_K).to(out_dtype)


def dequantize_q8_0_torch(blocks, out_dtype=torch.float16):
    """blocks [R, Nb*34] u8 (f16 d + 32×i8) -> [R, Nb*32] fp. Exact."""
    R = blocks.shape[0]
    total = blocks.shape[1] // 34
    t = blocks.reshape(R, total, 34)
    d = t[..., 0:2].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)
    q = t[..., 2:34].to(torch.int8).to(torch.float32)
    out = d[..., None] * q
    return out.reshape(R, total * 32).to(out_dtype)


def q4k_gemv_torch(blocks, x):
    """Fallback: dequantize then dense. blocks [R, Nb*144] u8, x [B, K] -> [B, R]."""
    W = dequantize_q4_k_torch(blocks, out_dtype=x.dtype)
    return (x.to(torch.float32) @ W.to(torch.float32).t()).to(x.dtype)


def q6k_gemv_torch(blocks, x):
    """Fallback: dequantize then dense. blocks [R, Nb*210] u8, x [B, K] -> [B, R]."""
    W = dequantize_q6_k_torch(blocks, out_dtype=x.dtype)
    return (x.to(torch.float32) @ W.to(torch.float32).t()).to(x.dtype)


# ---------- fused Triton decode GEMV (weights stay packed) ----------
@triton.jit
def _s8(v):
    """Sign-extend a zero-extended u8 load to int8 range."""
    return tl.where(v >= 128, v - 256, v)


@triton.jit
def _q4k_gemv_kernel(W, D, DMIN, X, Y, R, B, K, NB, ROWB,
                     SX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid % R
    b = pid // R
    acc = 0.0
    offs = tl.arange(0, BLOCK)
    for sb in range(NB):
        d = tl.load(D + r * NB + sb).to(tl.float32)
        dmin = tl.load(DMIN + r * NB + sb).to(tl.float32)
        sbase = r * ROWB + sb * 144
        # 12 scale bytes for this super-block (unrolled per sub-block below)
        for j in range(8):
            if j < 4:
                sc = (tl.load(W + sbase + 4 + j).to(tl.int32) & 63).to(tl.float32)
                m = (tl.load(W + sbase + 4 + j + 4).to(tl.int32) & 63).to(tl.float32)
            else:
                a = tl.load(W + sbase + 4 + j + 4).to(tl.int32)
                b0 = tl.load(W + sbase + 4 + j - 4).to(tl.int32)
                c = tl.load(W + sbase + 4 + j).to(tl.int32)
                sc = ((a & 0x0F) | ((b0 >> 6) << 4)).to(tl.float32)
                m = ((a >> 4) | ((c >> 6) << 4)).to(tl.float32)
            chunk = j // 2
            qb = tl.load(W + sbase + 16 + chunk * 32 + offs).to(tl.int32)
            if j % 2 == 0:
                q = (qb & 0x0F).to(tl.float32)
            else:
                q = ((qb >> 4) & 0x0F).to(tl.float32)
            xv = tl.load(X + b * SX + sb * 256 + j * 32 + offs).to(tl.float32)
            acc += tl.sum((d * sc * q - dmin * m) * xv, 0)
    tl.store(Y + b * R + r, acc)


@triton.jit
def _q6k_gemv_kernel(W, D, X, Y, R, B, K, NB, ROWB,
                     SX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid % R
    b = pid // R
    acc = 0.0
    L = tl.arange(0, 32)
    is_ = L // 16
    for sb in range(NB):
        d = tl.load(D + r * NB + sb).to(tl.float32)
        base = r * ROWB + sb * 210
        for n in range(2):
            ql0 = tl.load(W + base + n * 64 + L).to(tl.int32)
            ql1 = tl.load(W + base + n * 64 + 32 + L).to(tl.int32)
            qh0 = tl.load(W + base + 128 + n * 32 + L).to(tl.int32)
            # int8 scales: zero-extended u8 load needs sign fix-up
            s0 = _s8(tl.load(W + base + 192 + n * 8 + 0).to(tl.int32))
            s1 = _s8(tl.load(W + base + 192 + n * 8 + 1).to(tl.int32))
            s2 = _s8(tl.load(W + base + 192 + n * 8 + 2).to(tl.int32))
            s3 = _s8(tl.load(W + base + 192 + n * 8 + 3).to(tl.int32))
            s4 = _s8(tl.load(W + base + 192 + n * 8 + 4).to(tl.int32))
            s5 = _s8(tl.load(W + base + 192 + n * 8 + 5).to(tl.int32))
            s6 = _s8(tl.load(W + base + 192 + n * 8 + 6).to(tl.int32))
            s7 = _s8(tl.load(W + base + 192 + n * 8 + 7).to(tl.int32))
            sa = tl.where(is_ == 0, s0, s1).to(tl.float32)
            sb_ = tl.where(is_ == 0, s2, s3).to(tl.float32)
            sc_ = tl.where(is_ == 0, s4, s5).to(tl.float32)
            sd_ = tl.where(is_ == 0, s6, s7).to(tl.float32)
            q1 = ((ql0 & 0x0F) | ((qh0 & 3) << 4)).to(tl.float32) - 32.0
            q2 = ((ql1 & 0x0F) | (((qh0 >> 2) & 3) << 4)).to(tl.float32) - 32.0
            q3 = ((ql0 >> 4) | (((qh0 >> 4) & 3) << 4)).to(tl.float32) - 32.0
            q4 = ((ql1 >> 4) | (((qh0 >> 6) & 3) << 4)).to(tl.float32) - 32.0
            xoff = b * SX + sb * 256 + n * 128
            x0 = tl.load(X + xoff + L).to(tl.float32)
            x1 = tl.load(X + xoff + 32 + L).to(tl.float32)
            x2 = tl.load(X + xoff + 64 + L).to(tl.float32)
            x3 = tl.load(X + xoff + 96 + L).to(tl.float32)
            acc += tl.sum(d * sa * q1 * x0, 0)
            acc += tl.sum(d * sb_ * q2 * x1, 0)
            acc += tl.sum(d * sc_ * q3 * x2, 0)
            acc += tl.sum(d * sd_ * q4 * x3, 0)
    tl.store(Y + b * R + r, acc)


def _q4k_scales(blocks):
    """Per-block d/dmin fp32 from packed rows. blocks [R, Nb*144] -> [R, Nb]."""
    R = blocks.shape[0]
    t = blocks.reshape(R, -1, Q4K_BYTES)
    d = t[..., 0:2].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)
    dmin = t[..., 2:4].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)
    return d, dmin


def _q6k_scales(blocks):
    """Per-block d fp32 from packed rows. blocks [R, Nb*210] -> [R, Nb]."""
    R = blocks.shape[0]
    t = blocks.reshape(R, -1, Q6K_BYTES)
    d = t[..., 208:210].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)
    return d


def _q4k_triton(blocks, x, scales=None):
    B, K = x.shape
    R = blocks.shape[0]
    NB = K // SUB // 8
    if scales is None:
        d, dmin = _q4k_scales(blocks)
    else:
        d, dmin = scales
    Y = torch.empty(B, R, device=x.device, dtype=torch.float32)
    grid = (R * B,)
    _q4k_gemv_kernel[grid](blocks, d, dmin, x, Y, R, B, K, NB, NB * Q4K_BYTES,
                           x.stride(0), BLOCK=SUB)
    return Y.to(x.dtype)


def _q6k_triton(blocks, x, scales=None):
    B, K = x.shape
    R = blocks.shape[0]
    NB = K // SUB // 8
    d = _q6k_scales(blocks) if scales is None else scales
    Y = torch.empty(B, R, device=x.device, dtype=torch.float32)
    grid = (R * B,)
    _q6k_gemv_kernel[grid](blocks, d, x, Y, R, B, K, NB, NB * Q6K_BYTES,
                           x.stride(0), BLOCK=SUB)
    return Y.to(x.dtype)


def q4k_gemv(blocks, x, scales=None):
    """Fused Q4_K GEMV with torch fallback. blocks [R, Nb*144] packed u8,
    x [B, K] fp16/bf16 -> [B, R]. K must be a multiple of 256.
    scales: optional precomputed (d, dmin) from _q4k_scales (saves per-call
    temporaries on tight VRAM)."""
    if HAVE_TRITON and _cuda(x) and _cuda(blocks) and x.shape[1] % QK_K == 0:
        try:
            return _q4k_triton(blocks, x, scales)
        except Exception as exc:
            _note_fused_error("q4k", exc)
    return q4k_gemv_torch(blocks, x)


def q6k_gemv(blocks, x, scales=None):
    """Fused Q6_K GEMV with torch fallback. blocks [R, Nb*210] packed u8,
    x [B, K] fp16/bf16 -> [B, R]. K must be a multiple of 256."""
    import os as _os

    if _os.environ.get("MINICPM_TRACE"):
        print(f"q6k_gemv blocks={tuple(blocks.shape)} x={tuple(x.shape)}", flush=True)
    if HAVE_TRITON and _cuda(x) and _cuda(blocks) and x.shape[1] % QK_K == 0:
        try:
            return _q6k_triton(blocks, x, scales)
        except Exception as exc:
            _note_fused_error("q6k", exc)
    return q6k_gemv_torch(blocks, x)


def _note_fused_error(which, exc):
    import os as _os

    if _os.environ.get("MINICPM_DEBUG"):
        print(f"[minicpm_q4k] fused {which} fell back: {type(exc).__name__}: {exc}"[:300],
              flush=True)


# ---------- VRAM check ----------
def estimate_q4k_mb(n_q4_elems, n_q6_elems=0, n_f16_elems=0):
    """Packed weight footprint: 4.5 / 6.5625 / 16 bits per element."""
    bits = n_q4_elems * 4.5 + n_q6_elems * 6.5625 + n_f16_elems * 16.0
    return bits / 8.0 / (1024 ** 2)


# ---------- V1 gate: hand-computed exact vector ----------
def _hand_q4k_block():
    """One block with d=1, dmin=0, all sc=1, all m=0 -> output == nibbles.

    scales bytes [1,1,1,1, 0,0,0,0, 1,1,1,1] give sc=1,m=0 for every j:
    j<4 reads scales[j]&63=1 / scales[j+4]&63=0; j>=4 reads
    (scales[j+4]&0xF)|(scales[j-4]>>6<<4)=1 and (>>4)|(>>6<<4)=0.
    """
    blk = bytearray(Q4K_BYTES)
    blk[0:2] = b"\x00\x3c"  # d = 1.0 f16 LE
    blk[2:4] = b"\x00\x00"  # dmin = 0.0
    blk[4:16] = bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    blk[16:144] = bytes(range(128))
    return bytes(blk)


def _hand_q4k_expected():
    # qs bytes are range(128): chunk c holds bytes [c*32, c*32+32), low
    # nibbles -> weights [c*64, c*64+32), high nibbles -> [c*64+32, c*64+64).
    out = []
    for c in range(4):
        out.extend([(c * 32 + l) & 0xF for l in range(32)])
        out.extend([((c * 32 + l) >> 4) & 0xF for l in range(32)])
    return out


def _hand_q6k_block():
    """d=1.0, all int8 scales=1, ql=0x12, qh=0 -> q in {-30,-31} pattern."""
    blk = bytearray(Q6K_BYTES)
    blk[0:128] = bytes([0x12] * 128)
    blk[128:192] = bytes([0x00] * 64)
    blk[192:208] = bytes([0x01] * 16)
    blk[208:210] = b"\x00\x3c"  # d = 1.0 f16 LE
    return bytes(blk)


def _hand_q6k_expected():
    out = []
    for _ in range(2):
        out.extend([-30] * 32)
        out.extend([-30] * 32)
        out.extend([-31] * 32)
        out.extend([-31] * 32)
    return out


def test_hand_vector(device="cuda"):
    """V1 gate: exact hand-computed dequant. No tolerance games."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    ok = True
    b4 = torch.tensor(list(_hand_q4k_block()), dtype=torch.uint8, device=device).reshape(1, -1)
    got4 = dequantize_q4_k_torch(b4, out_dtype=torch.float32)[0].tolist()
    exp4 = [float(v) for v in _hand_q4k_expected()]
    err4 = max(abs(g - e) for g, e in zip(got4, exp4))
    print(f"  q4k hand vector: max_err={err4:.2e} {'PASS' if err4 == 0.0 else 'FAIL'}", flush=True)
    ok = ok and err4 == 0.0
    b6 = torch.tensor(list(_hand_q6k_block()), dtype=torch.uint8, device=device).reshape(1, -1)
    got6 = dequantize_q6_k_torch(b6, out_dtype=torch.float32)[0].tolist()
    exp6 = [float(v) for v in _hand_q6k_expected()]
    err6 = max(abs(g - e) for g, e in zip(got6, exp6))
    print(f"  q6k hand vector: max_err={err6:.2e} {'PASS' if err6 == 0.0 else 'FAIL'}", flush=True)
    return ok and err6 == 0.0


def test_against_torch(batch_size=2, R=256, K=512, rtol=2e-3):
    """V2 gate: fused GEMV vs torch dequant+matmul on random blocks.

    GEMV outputs are O(100) (random 4-bit weights x fp16 activations), so
    absolute atol is meaningless here — assert RELATIVE error, same spirit
    as the repo's 1e-3-class fp16 parity bars on normalized outputs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[minicpm_q4k.test] device={device} B={batch_size} R={R} K={K}", flush=True)
    if not test_hand_vector(device):
        print("  V1 hand vector FAIL", flush=True)
        return False
    torch.manual_seed(0)
    B = batch_size
    dtype = torch.float16
    x = (torch.randn(B, K, device=device, dtype=torch.float32) * 0.5).to(dtype)
    Nb = K // QK_K
    # NOTE: random block bytes can decode to NaN/Inf fp16 scales; pin the
    # embedded d/dmin to sane values (both paths read them from the blocks).
    def _pin_d(blk, per_block, vals):
        r = blk.reshape(R, Nb, per_block).clone()
        for off, v in vals:
            r[..., off:off + 2] = torch.full(
                (R, Nb, 1), v, dtype=torch.float16,
                device=device).view(torch.uint8).reshape(R, Nb, 2)
        return r.reshape(R, Nb * per_block)
    ok = True
    for name, nbytes, gemv, gemv_t in (
        ("q4k", Q4K_BYTES, _q4k_triton, q4k_gemv_torch),
        ("q6k", Q6K_BYTES, _q6k_triton, q6k_gemv_torch),
    ):
        blk = torch.randint(0, 256, (R, Nb * nbytes), device=device, dtype=torch.uint8)
        if name == "q4k":
            blk = _pin_d(blk, nbytes, ((0, 0.05), (2, 0.01)))
        else:
            blk = _pin_d(blk, nbytes, ((208, 0.05),))
        try:
            out_f = gemv(blk, x).float()
        except Exception as e:
            print(f"  {name} fused unavailable ({e}), torch-only", flush=True)
            continue
        out_t = gemv_t(blk, x).float()
        err = (out_f - out_t).abs().max().item()
        rel = err / max(out_t.abs().max().item(), 1e-9)
        good = rel < rtol
        print(f"  {name} parity: max_err={err:.2e} rel={rel:.2e} {'PASS' if good else 'FAIL'}", flush=True)
        ok = ok and good
    return ok


# ---------- perf_report inside model file ----------
try:
    import triton.testing
    _has_perf = True
except Exception:
    _has_perf = False

if _has_perf:
    _bench_list = [
        triton.testing.Benchmark(
            x_names=["B"],
            x_vals=[1, 2, 4, 8],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton fused Q4_K GEMV", "Torch dequant+mm"],
            styles=[("blue", "-"), ("orange", "--")],
            ylabel="ms",
            plot_name="minicpm-q4k-gemv-B",
            args={"R": 2048, "K": 2048},
        ),
    ]

    @triton.testing.perf_report(_bench_list)
    def bench_minicpm_q4k(B, R=2048, K=2048, provider="triton"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16
        torch.manual_seed(0)
        x = (torch.randn(B, K, device=device, dtype=torch.float32) * 0.5).to(dtype)
        Nb = K // QK_K
        blk = torch.randint(0, 256, (R, Nb * Q4K_BYTES), device=device, dtype=torch.uint8)
        import triton.testing as tt
        if provider == "triton":
            if not (HAVE_TRITON and _cuda(x)):
                return float("nan"), float("nan"), float("nan")
            fn = lambda: _q4k_triton(blk, x)  # noqa: E731
        else:
            fn = lambda: q4k_gemv_torch(blk, x)  # noqa: E731
        ms = tt.do_bench(fn, warmup=10, rep=30)
        return ms, ms * 0.9, ms * 1.1

    # To generate plots: run `python -m src.models.triton_kernels.minicpm_q4k --bench`
    if __name__ == "__main__":
        import argparse, os
        p = argparse.ArgumentParser()
        p.add_argument("--bench", action="store_true", help="run perf_report and save plots")
        p.add_argument("--save_path", default="benchmarks/results", help="where to save plots")
        p.add_argument("--test", action="store_true", help="run V1+V2 parity gates")
        args = p.parse_args()
        if args.test:
            raise SystemExit(0 if test_against_torch() else 1)
        if args.bench:
            os.makedirs(args.save_path, exist_ok=True)
            bench_minicpm_q4k.run(save_path=args.save_path, print_data=True)
            print(f"saved plots to {args.save_path}/minicpm-q4k-gemv-*.png")

HAVE_TRITON_KERNELS = HAVE_TRITON
__all__ = ["QK_K", "Q4K_BYTES", "Q6K_BYTES", "get_scale_min_k4",
           "dequantize_q4_k_torch", "dequantize_q6_k_torch",
           "dequantize_q8_0_torch",
           "q4k_gemv", "q6k_gemv", "q4k_gemv_torch", "q6k_gemv_torch",
           "estimate_q4k_mb", "test_hand_vector", "test_against_torch",
           "bench_minicpm_q4k"]
