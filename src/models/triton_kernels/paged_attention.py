"""Fused paged attention kernels — batched GQA with block-table indirection.

This kernel does one launch for all query heads across all sequences in a batch,
reading K/V from paged cache via block tables. Replaces per-position Python loops.

Shapes:
- Q: [B, Hq, Dh]   (batched queries for decode)
- K_cache: [num_blocks, block_size, Hk, Dh]  (paged K cache per layer)
- V_cache: [num_blocks, block_size, Hk, Dh]  (paged V cache per layer)
- block_tables: [B, max_blocks]  (physical block indices per seq)

Output: O [B, Hq, Dh]
"""
import torch
import triton
import triton.language as tl

# ---------- Paged GQA Decode Kernel ----------
@triton.jit
def _paged_gqa_decode_kernel(
    Q,                  # [B, Hq, Dh]
    K_cache,            # [num_blocks, block_size, Hk, Dh]
    V_cache,            # [num_blocks, block_size, Hk, Dh]
    O,                  # [B, Hq, Dh]
    block_tables,       # [B, max_blocks] int32
    seq_lens,           # [B] int32  (valid K/V length per seq)
    stride_qb, stride_qh, stride_qd,
    stride_k_blk, stride_k_slot, stride_k_h, stride_k_d,
    stride_v_blk, stride_v_slot, stride_v_h, stride_v_d,
    stride_ob, stride_oh, stride_od,
    B, Hq, Hk, Dh, scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (batch, query_head). Iterates over blocks in block_table."""
    # pid = b * Hq + hid
    pid = tl.program_id(0)
    b = pid // Hq
    hid = pid % Hq
    khid = hid // (Hq // Hk)  # GQA group

    # load query
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q + b * stride_qb + hid * stride_qh + offs_d).to(tl.float32)

    # online softmax state
    m = float("-inf")
    l = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # iterate over blocks in block_table
    seq_len = seq_lens[b]
    blocks_needed = (seq_len + 15) // 16  # block_size=16
    # we'll loop over blocks, each block has 16 tokens
    for blk_idx in range(blocks_needed):
        # get physical block id from block_table
        bid = tl.load(block_tables + b * 64 + blk_idx)  # max 64 blocks per seq
        if bid < 0:
            break

        # iterate over slots in this block (16 tokens)
        # we need to loop over slots and accumulate
        for slot in range(16):
            abs_pos = blk_idx * 16 + slot
            if abs_pos >= seq_len:
                break

            # load K from this block, slot, kv_head
            k = tl.load(
                K_cache + bid * stride_k_blk + slot * stride_k_slot + khid * stride_k_h + offs_d,
                mask=offs_d < Dh, other=0.0
            ).to(tl.float32)

            # compute score
            s = tl.sum(q * k) * scale

            # online softmax update
            m_new = tl.maximum(m, s)
            alpha = tl.exp(m - m_new)
            probs = tl.exp(s - m_new)
            l = l * alpha + probs

            # load V
            v = tl.load(
                V_cache + bid * stride_v_blk + slot * stride_v_slot + khid * stride_v_h + offs_d,
                mask=offs_d < Dh, other=0.0
            ).to(tl.float32)

            acc = acc * alpha + probs * v
            m = m_new

    # finalize
    acc = acc / l
    tl.store(O + b * stride_ob + hid * stride_oh + offs_d, acc, mask=offs_d < Dh)


def paged_gqa_decode(q, k_cache, v_cache, block_tables, seq_lens, scale, block_size=16):
    """Batched paged GQA decode.

    q: [B, Hq, Dh]
    k_cache: [num_blocks, block_size, Hk, Dh]
    v_cache: [num_blocks, block_size, Hk, Dh]
    block_tables: [B, max_blocks] int32 (padded with -1)
    seq_lens: [B] int32
    scale: float

    Returns: [B, Hq, Dh]
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    B, Hq, Dh = q.shape
    Hk = k_cache.shape[2]
    assert Hq % Hk == 0

    # q: [B, Hq, Dh] -> contiguous
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.contiguous()
    seq_lens = seq_lens.contiguous()

    O = torch.empty_like(q)
    B, Hq, Dh = q.shape
    Hk = k_cache.shape[2]
    assert Hq % Hk == 0

    BLOCK_D = 128
    grid = (B * Hq,)

    _paged_gqa_decode_kernel[grid](
        q, k_cache, v_cache, O,
        block_tables, seq_lens,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        B, Hq, Hk, Dh, scale,
        16,  # BLOCK_N
        128, # BLOCK_D
    )
    return O


# ---------- Batched Paged GQA for Prefill (multiple queries per seq) ----------
@triton.jit
def _paged_gqa_prefill_kernel(
    Q,                  # [B, Hq, Dh]  - multiple queries per seq (prefill chunk)
    K_cache,            # [num_blocks, block_size, Hk, Dh]
    V_cache,            # [num_blocks, block_size, Hk, Dh]
    O,                  # [B, Hq, Dh]
    block_tables,       # [B, max_blocks]
    seq_lens,           # [B] - total K/V length including history
    chunk_starts,       # [B] - start pos in K/V for each seq's chunk
    chunk_lens,         # [B] - number of queries per seq
    stride_qb, stride_qh, stride_qd,
    stride_k_blk, stride_k_slot, stride_k_h, stride_k_d,
    stride_v_blk, stride_v_slot, stride_v_h, stride_v_d,
    stride_ob, stride_oh, stride_od,
    B, Hq, Hk, Dh, scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (batch, query_head). Processes prefill chunk."""
    pid = tl.program_id(0)
    b = pid // Hq
    hid = pid % Hq
    khid = hid // (Hq // Hk)

    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q + b * stride_qb + hid * stride_qh + offs_d).to(tl.float32)

    m = float("-inf")
    l = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    seq_len = seq_lens[b]
    chunk_start = chunk_starts[b]
    chunk_len = chunk_lens[b]

    if chunk_len == 0:
        tl.store(O + b * stride_ob + hid * stride_oh + offs_d, acc, mask=offs_d < Dh)
        return

    # iterate over all K/V up to seq_len
    blocks_needed = (seq_len + 15) // 16
    for blk_idx in range(blocks_needed):
        bid = tl.load(block_tables + b * 64 + blk_idx)
        if bid < 0:
            break

        for slot in range(16):
            abs_pos = blk_idx * 16 + slot
            if abs_pos >= seq_len:
                break

            k = tl.load(
                K_cache + bid * stride_k_blk + slot * stride_k_slot + khid * stride_k_h + offs_d,
                mask=offs_d < Dh, other=0.0
            ).to(tl.float32)

            s = tl.sum(q * k) * scale
            m_new = tl.maximum(m, s)
            alpha = tl.exp(m - m_new)
            probs = tl.exp(s - m_new)
            l = l * alpha + probs

            v = tl.load(
                V_cache + bid * stride_v_blk + slot * stride_v_slot + khid * stride_v_h + offs_d,
                mask=offs_d < Dh, other=0.0
            ).to(tl.float32)

            acc = acc * alpha + probs * v
            m = m_new

    acc = acc / l
    tl.store(O + b * stride_ob + hid * stride_oh + offs_d, acc, mask=offs_d < Dh)


def paged_gqa_prefill(q, k_cache, v_cache, block_tables, seq_lens, chunk_starts, chunk_lens, scale, block_size=16):
    """Batched paged GQA prefill for chunked prefill.

    q: [B, Hq, Dh]  - one query per seq (prefill chunk)
    Returns: [B, Hq, Dh]
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    B, Hq, Dh = q.shape
    Hk = k_cache.shape[2]
    assert Hq % Hk == 0

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.contiguous()
    seq_lens = seq_lens.contiguous()
    chunk_starts = chunk_starts.contiguous()
    chunk_lens = chunk_lens.contiguous()

    O = torch.empty_like(q)
    B, Hq, Dh = q.shape
    Hk = k_cache.shape[2]
    assert Hq % Hk == 0

    BLOCK_D = 128
    grid = (B * Hq,)

    _paged_gqa_prefill_kernel[grid](
        q, k_cache, v_cache, O,
        block_tables, seq_lens, chunk_starts, chunk_lens,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        B, Hq, Hk, Dh, scale,
        16, 128,
    )
    return O


# ---------- Python fallbacks for CPU ----------
def paged_gqa_decode_torch(q, k_cache, v_cache, block_tables, seq_lens, scale):
    """CPU fallback for paged GQA decode."""
    B, Hq, Dh = q.shape
    Hk = k_cache.shape[2]
    group = Hq // Hk
    O = torch.zeros_like(q)
    for b in range(B):
        seq_len = seq_lens[b].item()
        table = block_tables[b]
        for hid in range(Hq):
            khid = hid // group
            m = float("-inf")
            l = 0.0
            acc = torch.zeros(Dh, dtype=torch.float32)
            q_head = q[b, hid].float()
            for blk_idx, bid in enumerate(table):
                if bid < 0:
                    break
                for slot in range(16):
                    abs_pos = blk_idx * 16 + slot
                    if abs_pos >= seq_len:
                        break
                    k = k_cache[bid, slot, khid].float()
                    v = v_cache[bid, slot, khid].float()
                    s = (q_head * k).sum() * scale
                    m_new = max(m, s)
                    alpha = math.exp(m - m_new)
                    probs = math.exp(s - m_new)
                    l = l * alpha + probs
                    acc = acc * alpha + probs * v
                    m = m_new
            acc = acc / l
            O[b, hid] = acc
    return O


import math