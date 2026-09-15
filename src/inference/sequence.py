"""Sequence / SequenceGroup — the scheduling unit."""

from __future__ import annotations

import time
import enum
from dataclasses import dataclass, field

from src.inference.config import SamplingParams

__all__ = [
    "SequenceStatus",
    "Sequence",
    "SequenceGroup",
]


class SequenceStatus(str, enum.Enum):
    WAITING = "WAITING"  # not yet admitted to running
    RUNNING = "RUNNING"  # has blocks, in batch
    SWAPPED = "SWAPPED"  # evicted, blocks freed (could be swapped to CPU)
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"


@dataclass
class Sequence:
    """Single token sequence with paged block table."""

    seq_id: int
    prompt_token_ids: list[int]
    block_size: int = 16
    status: SequenceStatus = SequenceStatus.WAITING
    # output tokens generated so far (excluding prompt)
    output_token_ids: list[int] = field(default_factory=list)
    # logical block table -> physical block ids
    block_table: list[int] = field(default_factory=list)
    # number of tokens already computed (prefill progress)
    num_computed_tokens: int = 0
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = field(default_factory=time.time)
    # metrics
    prompt_len: int = 0

    def __post_init__(self):
        self.prompt_len = len(self.prompt_token_ids)

    # -- token view ---------------------------------------------------
    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_token_ids(self) -> list[int]:
        return self.all_token_ids

    def get_len(self) -> int:
        return self.num_tokens

    def get_uncomputed_tokens(self) -> list[int]:
        # tokens not yet prefetched into KV
        all_ids = self.all_token_ids
        return all_ids[self.num_computed_tokens :]

    def get_num_uncomputed(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    # -- block accounting ---------------------------------------------
    def num_blocks_needed(self, block_size: int | None = None) -> int:
        bs = block_size or self.block_size
        n = self.num_tokens
        return (n + bs - 1) // bs

    def num_blocks_computed(self) -> int:
        # how many blocks are already allocated for computed prefix
        bs = self.block_size
        return (self.num_computed_tokens + bs - 1) // bs if self.num_computed_tokens else 0

    def append_token(self, token_id: int):
        self.output_token_ids.append(int(token_id))

    def is_finished(self) -> bool:
        if self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED):
            return True
        # stopped by length
        if len(self.output_token_ids) >= self.sampling_params.max_tokens:
            return True
        # stopped by EOS (check last token)
        if not self.sampling_params.ignore_eos and self.output_token_ids:
            if self.output_token_ids[-1] in self.sampling_params.stop_token_ids:
                return True
        return False

    def check_finished(self) -> bool:
        if self.is_finished():
            self.status = SequenceStatus.FINISHED
            return True
        return False


@dataclass
class SequenceGroup:
    """One user request = one SequenceGroup (single seq for now, supports beam)."""

    request_id: str
    sequences: list[Sequence]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.time)
    # prompt string for debug
    prompt: str | None = None

    @property
    def seq(self) -> Sequence:
        return self.sequences[0]

    def is_finished(self) -> bool:
        return all(s.status == SequenceStatus.FINISHED for s in self.sequences)

    def is_waiting(self) -> bool:
        return any(s.status == SequenceStatus.WAITING for s in self.sequences)

    def num_seqs(self, status: SequenceStatus | None = None) -> int:
        if status is None:
            return len(self.sequences)
        return sum(1 for s in self.sequences if s.status == status)

    def get_seqs(self, status: SequenceStatus | None = None) -> list[Sequence]:
        if status is None:
            return list(self.sequences)
        return [s for s in self.sequences if s.status == status]

    @classmethod
    def from_prompt(
        cls,
        request_id: str,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        block_size: int = 16,
        seq_id: int = 0,
    ) -> "SequenceGroup":
        sp = sampling_params or SamplingParams()
        seq = Sequence(
            seq_id=seq_id,
            prompt_token_ids=list(token_ids),
            block_size=block_size,
            status=SequenceStatus.WAITING,
            sampling_params=sp,
        )
        return cls(request_id=request_id, sequences=[seq], sampling_params=sp)
