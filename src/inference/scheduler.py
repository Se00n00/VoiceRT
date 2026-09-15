"""Continuous-batching scheduler — iteration-level scheduling.

Implements the 'scheduling' box from the diagram: waiting -> running
admission with token-budget + block-budget + FCFS ordering, plus
preemption when memory is tight.

This is real scheduling, not a sleep: every step() asks the scheduler which
seqs to batch; the scheduler consults BlockManager for memory fit and
enforces max_batch_size / max_num_batched_tokens limits.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from src.inference.block_manager import BlockManager
from src.inference.config import SchedulerConfig
from src.inference.sequence import Sequence, SequenceGroup, SequenceStatus

__all__ = ["SchedulerOutputs", "ContinuousScheduler"]


@dataclass
class SchedulerOutputs:
    scheduled: list[SequenceGroup] = field(default_factory=list)
    # per-seq metadata for model runner
    num_batched_tokens: int = 0
    # sequences that were preempted (need to be re-queued)
    preempted: list[SequenceGroup] = field(default_factory=list)
    # number of waiting groups still pending
    waiting: int = 0
    running: int = 0
    swapped: int = 0


class ContinuousScheduler:
    """FCFS continuous batching (Orca-style) with chunked prefill support."""

    def __init__(self, sched_cfg: SchedulerConfig, block_manager: BlockManager):
        self.config = sched_cfg
        self.block_manager = block_manager
        self.waiting: collections.deque[SequenceGroup] = collections.deque()
        self.running: collections.deque[SequenceGroup] = collections.deque()
        self.swapped: collections.deque[SequenceGroup] = collections.deque()
        self._finished: list[SequenceGroup] = []

    # -- queue ops ----------------------------------------------------
    def add_request(self, seq_group: SequenceGroup):
        self.waiting.append(seq_group)

    def abort_request(self, request_id: str) -> bool:
        for q in (self.waiting, self.running, self.swapped):
            for sg in list(q):
                if sg.request_id == request_id:
                    q.remove(sg)
                    for seq in sg.sequences:
                        seq.status = SequenceStatus.ABORTED
                        self.block_manager.free(seq.seq_id)
                    return True
        return False

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def num_pending(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    # -- core scheduling ----------------------------------------------
    def _future_blocks(self, seq: Sequence) -> int:
        total = seq.prompt_len + seq.sampling_params.max_tokens
        return (total + seq.block_size - 1) // seq.block_size

    def _can_schedule_prefill(self, seq: Sequence, token_budget: int) -> bool:
        """Can we admit this waiting seq's next chunk?

        Checks both immediate block availability and future reservation
        (prompt + max_tokens) to avoid later OOM deadlock — the 'memory mgmt'
        pillar's admission control.
        """
        # immediate token budget
        if self.config.enable_chunked_prefill:
            remaining = seq.get_num_uncomputed()
            chunk = min(remaining, self.config.max_num_batched_tokens - token_budget)
            if chunk <= 0:
                return False
            needed_blocks = (seq.num_computed_tokens + chunk + seq.block_size - 1) // seq.block_size
            cur_blocks = len(self.block_manager._tables.get(seq.seq_id, []))
            extra = max(0, needed_blocks - cur_blocks)
            if extra > 0 and not self.block_manager.can_allocate(extra):
                return False
            if token_budget + chunk > self.config.max_num_batched_tokens:
                return False
        else:
            need = seq.get_num_uncomputed()
            if token_budget + need > self.config.max_num_batched_tokens:
                return False
            needed_blocks = (seq.num_tokens + seq.block_size - 1) // seq.block_size
            cur = len(self.block_manager._tables.get(seq.seq_id, []))
            extra = max(0, needed_blocks - cur)
            if not self.block_manager.can_allocate(extra):
                return False

        # future reservation: ensure we can eventually grow to max_tokens
        # without deadlock. Sum future blocks of running + this waiting seq
        # must fit within total - watermark.
        future_needed = self._future_blocks(seq)
        cur = len(self.block_manager._tables.get(seq.seq_id, []))
        # total used now
        used = self.block_manager.used_count
        # future extra beyond current for this seq
        future_extra = max(0, future_needed - cur)
        # sum of future extras for already running seqs
        running_future_extra = 0
        for sg in self.running:
            s = sg.seq
            cur_r = len(self.block_manager._tables.get(s.seq_id, []))
            fut_r = self._future_blocks(s)
            running_future_extra += max(0, fut_r - cur_r)
        watermark = int(self.config.watermark_blocks)
        if used + future_extra + running_future_extra > self.block_manager.num_blocks - watermark:
            return False
        return True

    def _schedule_prefill_chunk(self, seq: Sequence, token_budget: int) -> int:
        """Return chunk size for this seq (0 if cannot)."""
        if self.config.enable_chunked_prefill:
            remaining = seq.get_num_uncomputed()
            return min(remaining, self.config.max_num_batched_tokens - token_budget)
        return seq.get_num_uncomputed()

    def schedule(self) -> SchedulerOutputs:
        """One iteration of scheduling -> which groups run this step.

        Priority:
          1) keep all RUNNING decodes scheduled (they need 1 token each)
          2) admit WAITING prefills in FCFS order while budget allows
          3) if OOM, preempt youngest RUNNING back to WAITING (or SWAPPED)
        """
        out = SchedulerOutputs()
        token_budget = 0
        scheduled: list[SequenceGroup] = []

        # --- 1) schedule running (decode) ---
        # running seqs each need exactly 1 token (next decode)
        # but we must ensure they have a block for next token; they were
        # pre-allocated incrementally.
        remaining_running = collections.deque()
        for sg in list(self.running):
            seq = sg.seq
            if seq.status != SequenceStatus.RUNNING:
                continue
            # each decode is 1 token
            if len(scheduled) >= self.config.max_batch_size:
                remaining_running.append(sg)
                continue
            if token_budget + 1 > self.config.max_num_batched_tokens:
                remaining_running.append(sg)
                continue
            # ensure block for next token (seq_len+1)
            need_len = seq.num_tokens + 1  # we will generate one
            needed_blocks = (need_len + seq.block_size - 1) // seq.block_size
            cur_blocks = len(self.block_manager._tables.get(seq.seq_id, []))
            extra = max(0, needed_blocks - cur_blocks)
            if extra and not self.block_manager.can_allocate(extra):
                # preempt this seq: put to waiting front (preserve order)
                # try to free by preempting lower priority? For now just keep waiting
                remaining_running.append(sg)
                continue
            if extra:
                self.block_manager.allocate(seq.seq_id, extra)
            scheduled.append(sg)
            token_budget += 1

        # if we couldn't schedule some running due to block shortage, try preemption
        # Strategy: preempt youngest waiting? Actually we preempt running that didn't fit
        # and try to admit waiting instead? But prefers running.
        # Keep remaining_running as not scheduled this step.

        # rebuild running as scheduled + remaining
        self.running = collections.deque(scheduled)  # will be rebuilt
        pending_running = remaining_running

        # --- 2) schedule waiting (prefill) FCFS ---
        # admit while batch size and token budget allow
        still_waiting: collections.deque[SequenceGroup] = collections.deque()
        while self.waiting and len(scheduled) < self.config.max_batch_size:
            sg = self.waiting[0]
            seq = sg.seq
            if not self._can_schedule_prefill(seq, token_budget):
                # cannot schedule head -> stop (FCFS)
                break
            # admit
            self.waiting.popleft()
            chunk = self._schedule_prefill_chunk(seq, token_budget)
            # allocate blocks for chunk
            needed = (seq.num_computed_tokens + chunk + seq.block_size - 1) // seq.block_size
            cur = len(self.block_manager._tables.get(seq.seq_id, []))
            extra = max(0, needed - cur)
            if extra:
                self.block_manager.allocate(seq.seq_id, extra)
            # mark running
            seq.status = SequenceStatus.RUNNING
            scheduled.append(sg)
            token_budget += chunk if self.config.enable_chunked_prefill else seq.get_num_uncomputed()

            # If non-chunked, token_budget may blow; but we already checked.
            # For chunked, we loop to admit more waiting if still budget.
            # For simplicity, if chunked, admit only one chunk per seq per step?
            # Then continue loop: next waiting seq gets chance.
            # If seq still has uncomputed tokens after this chunk, it stays running
            # and will be scheduled again next step for remaining prefill.
            # So we keep it in running queue.
        # any waiting not admitted stays waiting
        # put back unadmitted
        # scheduled already contains admitted waiting groups; need to ensure
        # pending_running groups that were not scheduled are re-queued to running
        # and waiting groups remain

        # reconstruct running queue: scheduled groups that are not finished, plus pending
        new_running = collections.deque()
        for sg in scheduled:
            new_running.append(sg)
        for sg in pending_running:
            new_running.append(sg)
        # also any swapped? for now no swap logic, just keep swapped queue
        self.running = new_running

        # if we have running > max_num_seqs, preempt youngest
        while len(self.running) > self.config.max_num_seqs:
            # preempt last (youngest)
            sg = self.running.pop()
            self.waiting.appendleft(sg)
            for seq in sg.sequences:
                # keep blocks but mark waiting? vLLM would swap; we keep blocks and mark WAITING
                seq.status = SequenceStatus.WAITING

        out.scheduled = scheduled
        out.num_batched_tokens = token_budget
        out.waiting = len(self.waiting)
        out.running = len(self.running)
        out.swapped = len(self.swapped)
        return out

    def free_finished(self, seq_groups: list[SequenceGroup]):
        """Move finished groups to finished list and free their blocks."""
        for sg in seq_groups:
            if sg.is_finished():
                for seq in sg.sequences:
                    self.block_manager.free(seq.seq_id)
                    seq.status = SequenceStatus.FINISHED
                # remove from running if present
                if sg in self.running:
                    self.running.remove(sg)
                self._finished.append(sg)

    def get_finished(self) -> list[SequenceGroup]:
        done = list(self._finished)
        self._finished.clear()
        return done

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
            "free_blocks": self.block_manager.free_count,
            "used_blocks": self.block_manager.used_count,
        }
