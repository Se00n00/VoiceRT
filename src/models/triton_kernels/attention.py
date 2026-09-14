"""Decode attention kernels (MHA + GQA, single query over KV-cache)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _dec_attn_kernel(Q, K, V, O,
                     stride_qh, stride_kh, stride_kn, stride_vh, stride_vn,
                     N, scale,
                     D: tl.constexpr, BLOCK_N: tl.constexpr):
    hid = tl.program_id(0)  # one program per head
    offs_d = tl.arange(0, D)
    q = tl.load(Q + hid * stride_qh + offs_d).to(tl.float32)
    m = float("-inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(K + hid * stride_kh + offs_n[:, None] * stride_kn +
                    offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, 1) * scale
        s = tl.where(offs_n < N, s, float("-inf"))  # masked lanes must not
        m_new = tl.maximum(m, tl.max(s, 0))         # pollute max/sum
        alpha = tl.exp(m - m_new)
        probs = tl.exp(s - m_new)
        l = l * alpha + tl.sum(probs, 0)
        v = tl.load(V + hid * stride_vh + offs_n[:, None] * stride_vn +
                    offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * v, 0)
        m = m_new
    acc = acc / l
    tl.store(O + hid * stride_qh + offs_d, acc)


@triton.jit
def _bdec_attn_kernel(Q, K, V, O,
                      stride_qb, stride_kb, stride_kn, stride_vb, stride_vn,
                      N, scale,
                      D: tl.constexpr, BLOCK_N: tl.constexpr):
    bhid = tl.program_id(0)  # flattened batch*heads
    offs_d = tl.arange(0, D)
    # NOTE: caller flattens (B,H) into grid; strides passed per-tensor
    q = tl.load(Q + bhid * stride_qb + offs_d).to(tl.float32)
    m = float("-inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(K + bhid * stride_kb + offs_n[:, None] * stride_kn +
                    offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, 1) * scale
        s = tl.where(offs_n < N, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 0))
        alpha = tl.exp(m - m_new)
        probs = tl.exp(s - m_new)
        l = l * alpha + tl.sum(probs, 0)
        v = tl.load(V + bhid * stride_vb + offs_n[:, None] * stride_vn +
                    offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * v, 0)
        m = m_new
    acc = acc / l
    tl.store(O + bhid * stride_qb + offs_d, acc)


@triton.jit
def _gqa_dec_kernel(Q, K, V, O,
                    stride_qh, stride_kh, stride_kn, stride_vh, stride_vn,
                    N, scale, group,
                    D: tl.constexpr, BLOCK_N: tl.constexpr):
    # one program per QUERY head; kv head = hid // group (GQA/MHA unified)
    hid = tl.program_id(0)
    khid = hid // group
    offs_d = tl.arange(0, D)
    q = tl.load(Q + hid * stride_qh + offs_d).to(tl.float32)
    m = float("-inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(K + khid * stride_kh + offs_n[:, None] * stride_kn +
                    offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, 1) * scale
        s = tl.where(offs_n < N, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 0))
        alpha = tl.exp(m - m_new)
        probs = tl.exp(s - m_new)
        l = l * alpha + tl.sum(probs, 0)
        v = tl.load(V + khid * stride_vh + offs_n[:, None] * stride_vn +
                    offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * v, 0)
        m = m_new
    acc = acc / l
    tl.store(O + hid * stride_qh + offs_d, acc)


def decode_attn(q, K, V, scale):
    """q [H,D], K/V [H,N,D] (strides free) -> O [H,D]. N runtime-variable."""
    assert q.is_cuda and K.is_cuda and V.is_cuda
    H, D = q.shape
    N = K.shape[-2]
    O = torch.empty_like(q)
    BLOCK_N = 128
    grid = (H,)
    _dec_attn_kernel[grid](q, K, V, O,
                           q.stride(0), K.stride(0), K.stride(1),
                           V.stride(0), V.stride(1),
                           N, scale, D, BLOCK_N)
    return O


def batched_decode_attn(q, K, V, scale):
    """q [B,H,D], K/V [B,H,N,D] (strides free) -> O [B,H,D]. One launch.

    Strides are read off the reshaped tensors, NOT assumed compact: reshape
    may return a strided view (e.g. over cache slices), and hardcoded strides
    silently misaddress every head but the first.
    """
    assert q.is_cuda and K.is_cuda and V.is_cuda
    B, H, D = q.shape
    N = K.shape[-2]
    qr = q.reshape(B * H, D)
    Kr = K.reshape(B * H, N, D)
    Vr = V.reshape(B * H, N, D)
    Or = torch.empty((B * H, D), device=q.device, dtype=q.dtype)
    BLOCK_N = 128
    grid = (B * H,)
    _bdec_attn_kernel[grid](qr, Kr, Vr, Or,
                            qr.stride(0), Kr.stride(0), Kr.stride(1),
                            Vr.stride(0), Vr.stride(1),
                            N, scale, D, BLOCK_N)
    return Or.reshape(B, H, D)


def gqa_decode_attn(q, K, V, scale):
    """q [Hq,D], K/V [Hk,N,D] (strides free) -> O [Hq,D]. Hq % Hk == 0."""
    assert q.is_cuda and K.is_cuda and V.is_cuda
    Hq, D = q.shape
    Hk, N = K.shape[0], K.shape[-2]
    assert Hq % Hk == 0
    qr = q.reshape(Hq, D)
    Kr = K.reshape(Hk, N, D)
    Vr = V.reshape(Hk, N, D)
    O = torch.empty((Hq, D), device=q.device, dtype=q.dtype)
    grid = (Hq,)
    _gqa_dec_kernel[grid](qr, Kr, Vr, O,
                          qr.stride(0), Kr.stride(0), Kr.stride(1),
                          Vr.stride(0), Vr.stride(1),
                          N, scale, Hq // Hk, D, 128)
    return O


@triton.jit
def _fused_qkv_kernel(
    X, Wq, Wk, Wv, Bq, Bk, Bv, Q, Kout, V,
    stride_wq0, stride_wq1,
    stride_wk0, stride_wk1,
    stride_wv0, stride_wv1,
    K_DIM, D,
    HAS_BQ: tl.constexpr, HAS_BK: tl.constexpr, HAS_BV: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused QKV GEMV for whisper decode (MHA, equal D).

    One program per BLOCK_N outputs; each program computes the same
    output tile for q, k and v (x loaded once, reused 3x).
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < D
    acc_q = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_k = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K_DIM, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_DIM
        x = tl.load(X + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        wq = tl.load(Wq + offs_n[:, None] * stride_wq0 + offs_k[None, :] * stride_wq1,
                     mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        wk = tl.load(Wk + offs_n[:, None] * stride_wk0 + offs_k[None, :] * stride_wk1,
                     mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        wv = tl.load(Wv + offs_n[:, None] * stride_wv0 + offs_k[None, :] * stride_wv1,
                     mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        acc_q += tl.sum(wq * x[None, :], 1)
        acc_k += tl.sum(wk * x[None, :], 1)
        acc_v += tl.sum(wv * x[None, :], 1)
    if HAS_BQ:
        bq = tl.load(Bq + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc_q = acc_q + bq
    if HAS_BK:
        bk = tl.load(Bk + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc_k = acc_k + bk
    if HAS_BV:
        bv = tl.load(Bv + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc_v = acc_v + bv
    tl.store(Q + offs_n, acc_q.to(Q.dtype.element_ty), mask=mask_n)
    tl.store(Kout + offs_n, acc_k.to(Kout.dtype.element_ty), mask=mask_n)
    tl.store(V + offs_n, acc_v.to(V.dtype.element_ty), mask=mask_n)


def fused_qkv(x, wq, wk, wv, bq=None, bv=None, bk=None):
    """Fused QKV GEMV for whisper decode: x [D] -> (q [D], k [D], v [D]).

    Convention VERIFIED from models/whisper/attention.py + encoder.py:
    q and v have bias, k has NO bias (F.linear(..., None) for k).
    ``bk`` is therefore optional (None); ``bq``/``bv`` may also be None
    for generic callers. One Triton launch producing all three slices.
    """
    assert x.is_cuda and wq.is_cuda and wk.is_cuda and wv.is_cuda
    assert x.dim() == 1
    assert wq.dim() == 2 and wk.dim() == 2 and wv.dim() == 2
    K_DIM = x.shape[0]
    D = wq.shape[0]
    assert wk.shape[0] == D and wv.shape[0] == D
    assert wq.shape[1] == K_DIM and wk.shape[1] == K_DIM and wv.shape[1] == K_DIM
    if bq is not None:
        assert bq.shape == (D,)
    if bv is not None:
        assert bv.shape == (D,)
    if bk is not None:
        assert bk.shape == (D,)
    xc = x.contiguous()
    wqc = wq.contiguous()
    wkc = wk.contiguous()
    wvc = wv.contiguous()
    dummy = xc
    Bq = bq.contiguous() if bq is not None else dummy
    Bk = bk.contiguous() if bk is not None else dummy
    Bv = bv.contiguous() if bv is not None else dummy
    Q = torch.empty((D,), device=x.device, dtype=x.dtype)
    Kout = torch.empty((D,), device=x.device, dtype=x.dtype)
    Vo = torch.empty((D,), device=x.device, dtype=x.dtype)
    BLOCK_N, BLOCK_K = 64, 64
    grid = (triton.cdiv(D, BLOCK_N),)
    _fused_qkv_kernel[grid](
        xc, wqc, wkc, wvc, Bq, Bk, Bv, Q, Kout, Vo,
        wqc.stride(0), wqc.stride(1),
        wkc.stride(0), wkc.stride(1),
        wvc.stride(0), wvc.stride(1),
        K_DIM, D,
        bq is not None, bk is not None, bv is not None,
        BLOCK_K, BLOCK_N,
    )
    return Q, Kout, Vo


@triton.jit
def _fused_qkv_gqa_kernel(
    X, Wq, Wk, Wv, Bq, Bk, Bv, Q, Kout, V,
    stride_wq0, stride_wq1,
    stride_wk0, stride_wk1,
    stride_wv0, stride_wv1,
    K_DIM, Dq, Dkv,
    HAS_BQ: tl.constexpr, HAS_BK: tl.constexpr, HAS_BV: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused QKV GEMV for GQA decode (Hq query heads, Hk kv heads).

    Grid covers Dq (query dim, the largest); kv tiles reuse the same
    pid with their own mask (inactive lanes produce zeros, never stored).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_q = offs < Dq
    mask_kv = offs < Dkv
    acc_q = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_k = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K_DIM, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_DIM
        x = tl.load(X + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        wq = tl.load(Wq + offs[:, None] * stride_wq0 + offs_k[None, :] * stride_wq1,
                     mask=mask_q[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        wk = tl.load(Wk + offs[:, None] * stride_wk0 + offs_k[None, :] * stride_wk1,
                     mask=mask_kv[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        wv = tl.load(Wv + offs[:, None] * stride_wv0 + offs_k[None, :] * stride_wv1,
                     mask=mask_kv[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        acc_q += tl.sum(wq * x[None, :], 1)
        acc_k += tl.sum(wk * x[None, :], 1)
        acc_v += tl.sum(wv * x[None, :], 1)
    if HAS_BQ:
        bq = tl.load(Bq + offs, mask=mask_q, other=0.0).to(tl.float32)
        acc_q = acc_q + bq
    if HAS_BK:
        bk = tl.load(Bk + offs, mask=mask_kv, other=0.0).to(tl.float32)
        acc_k = acc_k + bk
    if HAS_BV:
        bv = tl.load(Bv + offs, mask=mask_kv, other=0.0).to(tl.float32)
        acc_v = acc_v + bv
    tl.store(Q + offs, acc_q.to(Q.dtype.element_ty), mask=mask_q)
    tl.store(Kout + offs, acc_k.to(Kout.dtype.element_ty), mask=mask_kv)
    tl.store(V + offs, acc_v.to(V.dtype.element_ty), mask=mask_kv)


def fused_qkv_gqa(x, wq, wk, wv, bq=None, bk=None, bv=None):
    """Fused GQA QKV GEMV: x [K] -> (q [Dq], k [Dkv], v [Dkv]).

    Convention VERIFIED from models/qwen/attention.py: Qwen2.5-0.5B
    q/k/v projs ALL carry a bias (F.linear(..., w[...bias]) for each).
    All biases are optional here (None allowed) for generic callers.
    One Triton launch producing q, k, v slices.
    """
    assert x.is_cuda and wq.is_cuda and wk.is_cuda and wv.is_cuda
    assert x.dim() == 1
    assert wq.dim() == 2 and wk.dim() == 2 and wv.dim() == 2
    K_DIM = x.shape[0]
    Dq = wq.shape[0]
    Dkv = wk.shape[0]
    assert wv.shape[0] == Dkv
    assert wq.shape[1] == K_DIM and wk.shape[1] == K_DIM and wv.shape[1] == K_DIM
    if bq is not None:
        assert bq.shape == (Dq,)
    if bk is not None:
        assert bk.shape == (Dkv,)
    if bv is not None:
        assert bv.shape == (Dkv,)
    xc = x.contiguous()
    wqc = wq.contiguous()
    wkc = wk.contiguous()
    wvc = wv.contiguous()
    dummy = xc
    Bq = bq.contiguous() if bq is not None else dummy
    Bk = bk.contiguous() if bk is not None else dummy
    Bv = bv.contiguous() if bv is not None else dummy
    Q = torch.empty((Dq,), device=x.device, dtype=x.dtype)
    Kout = torch.empty((Dkv,), device=x.device, dtype=x.dtype)
    Vo = torch.empty((Dkv,), device=x.device, dtype=x.dtype)
    BLOCK_N, BLOCK_K = 64, 64
    grid = (triton.cdiv(Dq, BLOCK_N),)
    _fused_qkv_gqa_kernel[grid](
        xc, wqc, wkc, wvc, Bq, Bk, Bv, Q, Kout, Vo,
        wqc.stride(0), wqc.stride(1),
        wkc.stride(0), wkc.stride(1),
        wvc.stride(0), wvc.stride(1),
        K_DIM, Dq, Dkv,
        bq is not None, bk is not None, bv is not None,
        BLOCK_K, BLOCK_N,
    )
    return Q, Kout, Vo
