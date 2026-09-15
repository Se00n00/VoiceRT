"""Batch formation — the 'batching' box.

Takes SchedulerOutputs -> InputBatch for the model runner. Handles
padding, position ids, block tables and attention metadata.

Real batching: multiple sequences' tokens are concatenated and fed as one
forward; KV gather uses per-seq block tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.inference.sequence import SequenceGroup

__all__ = ["InputBatch", "make_batch"]


@dataclass
class InputBatch:
    """One forward batch."""

    # flattened tokens for this step: [num_batched_tokens]
    input_ids: torch.Tensor
    # positions: [num_batched_tokens] absolute pos per token
    positions: torch.Tensor
    # per-seq lengths before this step
    seq_lens: list[int]
    # per-seq block tables (list of lists)
    block_tables: list[list[int]]
    # which seq each token belongs to (for sampling)
    seq_ids: list[int]
    # per-seq uncomputed token counts (for prefill chunking)
    # also seq_groups in scheduled order
    seq_groups: list[SequenceGroup] = field(default_factory=list)
    # whether each seq is prefill vs decode
    is_prefill: list[bool] = field(default_factory=list)
    # number of tokens per seq in this batch
    num_tokens_per_seq: list[int] = field(default_factory=list)

    @property
    def num_seqs(self) -> int:
        return len(self.seq_groups)

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.numel()) if self.input_ids.numel() else 0

    @property
    def is_cuda(self) -> bool:
        return self.input_ids.is_cuda


def make_batch(
    scheduled: list[SequenceGroup],
    device: torch.device | str = "cpu",
    token_budget: int | None = None,
) -> InputBatch | None:
    """Build InputBatch from scheduled groups.

    For each seq:
      - if seq has uncomputed tokens -> prefill chunk (may be >1 token)
      - else -> decode (1 token = last token)
    Supports chunked prefill: if token_budget is set, prefill chunks are sliced
    to fit budget (at most `budget` tokens per batch).
    """
    if not scheduled:
        return None
    if isinstance(device, str):
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

    input_ids: list[int] = []
    positions: list[int] = []
    seq_lens: list[int] = []
    block_tables: list[list[int]] = []
    seq_ids: list[int] = []
    is_prefill: list[bool] = []
    num_tokens_per_seq: list[int] = []

    # token budget for chunked prefill
    remaining_budget = token_budget if token_budget is not None else 10**9

    for sg in scheduled:
        seq = sg.seq
        seq_lens.append(seq.num_tokens)  # before step
        block_tables.append([])  # placeholder, filled by engine
        remaining = seq.get_num_uncomputed()
        if remaining > 0:
            tokens = seq.get_uncomputed_tokens()
            # respect chunk budget
            if remaining_budget is not None:
                take = min(len(tokens), remaining_budget)
                tokens = tokens[:take]
                remaining_budget -= take
            start = seq.num_computed_tokens
            for i, tok in enumerate(tokens):
                input_ids.append(int(tok))
                positions.append(start + i)
                seq_ids.append(seq.seq_id)
            is_prefill.append(True)
            num_tokens_per_seq.append(len(tokens))
            if remaining_budget is not None and remaining_budget <= 0:
                # budget exhausted, remaining scheduled groups will be truncated to 1 token? Actually they were already scheduled by scheduler with budget, so this should not happen
                pass
        else:
            last = seq.all_token_ids[-1] if seq.all_token_ids else 0
            input_ids.append(int(last))
            positions.append(seq.num_tokens)
            positions[-1] = seq.num_tokens
            seq_ids.append(seq.seq_id)
            is_prefill.append(False)
            num_tokens_per_seq.append(1)
            if remaining_budget is not None:
                remaining_budget -= 1

    return InputBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        positions=torch.tensor(positions, dtype=torch.long, device=device),
        seq_lens=seq_lens,
        block_tables=block_tables,
        seq_ids=seq_ids,
        seq_groups=scheduled,
        is_prefill=is_prefill,
        num_tokens_per_seq=num_tokens_per_seq,
    )

