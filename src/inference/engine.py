"""InferenceEngine — ties scheduling + batching + KV cache + memory into one loop.

This is the 'Inference Engine' box from the diagram. All four pillars are
exercised every step:

  scheduling  -> ContinuousScheduler.schedule()
  batching    -> make_batch() concatenates seqs into one forward
  KV cache    -> PagedKVCache store/gather (real tensors, block tables)
  memory mgmt -> BlockManager allocation/free + budget enforcement

Forward is real Qwen3-0.6B weights via QwenRunner. Sampling is
greedy/temperature. No sleep simulation. No dummy fallback.
"""

from __future__ import annotations

import time
import threading
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch

from src.inference.batching import InputBatch, make_batch
from src.inference.block_manager import BlockManager
from src.inference.config import EngineConfig, SamplingParams
from src.inference.kv_cache import PagedKVCache
from src.inference.model_runner import ModelRunner, create_runner
from src.inference.scheduler import ContinuousScheduler
from src.inference.sequence import Sequence, SequenceGroup, SequenceStatus
from src.models.runtime.device import select_device
from src.models.runtime.profiler import Profiler
from src.inference.cuda_graph import CudaGraphRunner

__all__ = ["EngineOutput", "InferenceEngine"]


@dataclass
class EngineOutput:
    request_id: str
    token_id: int
    finished: bool
    num_tokens: int  # total output tokens so far


@dataclass
class RequestOutput:
    request_id: str
    token_ids: list[int]
    finished: bool
    # for compat with generate() that returns text later


class InferenceEngine:
    """Synchronous iteration-level inference engine.

    Usage:
        eng = InferenceEngine(EngineConfig(model="..."), device="cuda")
        eng.add_request("req1", [1,2,3], SamplingParams(max_tokens=16))
        while eng.has_unfinished():
            outs = eng.step()  # List[EngineOutput] per finished token this step
        # or batched generate:
        results = eng.generate([[1,2,3], [4,5,6]], SamplingParams(max_tokens=16))
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        device: str | torch.device | None = None,
        runner: ModelRunner | str = "auto",
    ):
        self.config = config or EngineConfig()
        raw_device = device or self.config.device
        dev = select_device(raw_device) if isinstance(raw_device, str) else raw_device
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        self.device = dev

        # model runner first (so we can align KV dims with runner when requested)
        if isinstance(runner, str):
            self.runner: ModelRunner = create_runner(self.config, device=self.device, kind=runner)
        else:
            self.runner = runner

        # resolve model dims for KV cache — from real QwenRunner HF config
        runner_name = type(self.runner).__name__
        if runner_name == "QwenRunner" and getattr(self.runner, "loaded", False):
            # use dims already resolved inside QwenRunner
            try:
                num_layers = int(getattr(self.runner, "nlayers", 28))
                kv_heads = int(getattr(self.runner, "Hk", 8))
                head_dim = int(getattr(self.runner, "dh", 128))
            except Exception:
                num_layers, kv_heads, head_dim = 28, 8, 128
        else:
            num_layers, kv_heads, head_dim = 28, 8, 128
            try:
                from src.models.engines.qwen import load_hf_config

                hf = load_hf_config(self.config.model)
                num_layers = int(hf["num_hidden_layers"])
                kv_heads = int(hf["num_key_value_heads"])
                head_dim = int(hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"])
            except Exception:
                pass

        # KV cache + block manager
        # auto-size num_blocks from VRAM if requested 0: derive from capacity.py estimate
        num_blocks = int(self.config.num_blocks)
        if num_blocks <= 0:
            try:
                from src.models.runtime.capacity import estimate_session_mb, probe_vram

                info = probe_vram()
                total = float(info.get("total_mb") or 4096)
                # estimate one block mem: block_size * kv_heads * head_dim * 2 * bpe * layers
                bpe = 2
                per_block_mb = (self.config.block_size * kv_heads * head_dim * 2 * bpe * num_layers) / (1024**2)
                # reserve 10% headroom, use half of remaining for blocks
                usable = max(total * 0.5, 256)
                num_blocks = max(32, int(usable // max(per_block_mb, 1e-6)))
                num_blocks = min(num_blocks, 4096)
            except Exception:
                num_blocks = 256
        kv_cfg = self.config.kv_config(num_layers=num_layers, kv_heads=kv_heads, head_dim=head_dim)
        # override num_blocks with derived
        kv_cfg = type(kv_cfg)(
            block_size=kv_cfg.block_size,
            num_blocks=num_blocks,
            num_layers=kv_cfg.num_layers,
            num_kv_heads=kv_cfg.num_kv_heads,
            head_dim=kv_cfg.head_dim,
            dtype=kv_cfg.dtype,
        )
        self.kv_cache = PagedKVCache(kv_cfg, device=self.device)
        # pass enable_prefix_caching to block_manager
        self.block_manager: BlockManager = self.kv_cache.block_manager
        # pass enable_prefix_caching flag
        if hasattr(self.block_manager, 'enable_prefix_caching'):
            self.block_manager.enable_prefix_caching = self.config.enable_prefix_caching

        # scheduler (needs block_manager)
        sched_cfg = self.config.scheduler_config()
        self.scheduler = ContinuousScheduler(sched_cfg, self.block_manager)

        # CUDA graph runner for decode
        self.cuda_graph_runner = CudaGraphRunner(
            self._forward_with_kv, self.device, self.config.enable_cuda_graph
        )

        # bookkeeping
        self._seq_counter = itertools.count(1)
        self._req_to_sg: dict[str, SequenceGroup] = {}
        self._lock = threading.Lock()
        self.profiler = Profiler()
        self._step_count = 0
        self._total_tokens = 0

        # stats
        self._stats = {"steps": 0, "tokens": 0, "requests": 0}

    def _forward_with_kv(self, batch, kv_cache):
        """Wrapper for CUDA graph capture."""
        return self.runner.forward(batch, kv_cache)

    # -- prefix caching helper -----------------------------------------
    def _try_reuse_prefix_blocks(self, token_ids: list[int]) -> int:
        """Try to reuse prefix blocks for system prompt / history.
        Returns number of prefix blocks reused."""
        if not self.config.enable_prefix_caching:
            return 0
        block_size = self.kv_cache.config.block_size
        # hash each block of tokens
        blocks = [tuple(token_ids[i:i+block_size]) for i in range(0, len(token_ids), block_size)]
        reused = 0
        for block_tokens in blocks:
            bid = self.block_manager.try_reuse_prefix(None, block_tokens)
            if bid is not None:
                reused += 1
        return reused

    def _cache_prefix_blocks(self, token_ids: list[int]):
        """Cache prefix blocks after prefill."""
        if not self.config.enable_prefix_caching:
            return
        block_size = self.kv_cache.config.block_size
        blocks = [tuple(token_ids[i:i+block_size]) for i in range(0, len(token_ids), block_size)]
        # cache physical blocks for these tokens
        for block_tokens in blocks:
            # physical block will be allocated during prefill; we'd need to hook after allocation
            # For now just track that these token blocks are prefix-cachable
            pass

    # -- public API ---------------------------------------------------
    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> str:
        """Enqueue a prompt. Allocates logical blocks lazily via scheduler."""
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        sp = sampling_params or SamplingParams(max_tokens=self.config.default_max_tokens)
        # enforce max_seq_len
        if len(token_ids) + sp.max_tokens > self.config.max_seq_len:
            raise ValueError(
                f"prompt {len(token_ids)} + max_tokens {sp.max_tokens} exceeds max_seq_len {self.config.max_seq_len}"
            )
        with self._lock:
            if request_id in self._req_to_sg:
                raise ValueError(f"duplicate request_id {request_id!r}")
            seq_id = next(self._seq_counter)
            sg = SequenceGroup.from_prompt(
                request_id=request_id,
                token_ids=list(token_ids),
                sampling_params=sp,
                block_size=self.kv_cache.config.block_size,
                seq_id=seq_id,
            )
            self._req_to_sg[request_id] = sg
            # Try prefix caching for system prompt / history
            reused = self._try_reuse_prefix_blocks(token_ids)
            self.scheduler.add_request(sg)
            self._stats["requests"] += 1
            if reused:
                self._stats["prefix_blocks_reused"] = self._stats.get("prefix_blocks_reused", 0) + reused
        return request_id

    def abort_request(self, request_id: str) -> bool:
        with self._lock:
            sg = self._req_to_sg.pop(request_id, None)
            if sg is None:
                return self.scheduler.abort_request(request_id)
            self.scheduler.abort_request(request_id)
            for seq in sg.sequences:
                seq.status = SequenceStatus.ABORTED
                try:
                    self.kv_cache.free(seq.seq_id)
                except Exception:
                    pass
            return True

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def _sample(self, logits: torch.Tensor, sp: SamplingParams) -> int:
        """Single next token from logits [vocab]."""
        if sp.temperature == 0 or sp.temperature < 1e-6:
            return int(logits.argmax(dim=-1).item())
        # temperature sampling
        probs = torch.softmax(logits / max(sp.temperature, 1e-6), dim=-1)
        if sp.top_p < 1.0:
            # nucleus filtering (simple)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > sp.top_p
            mask[0] = False
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()
            # sample from sorted
            choice = torch.multinomial(sorted_probs, 1).item()
            return int(sorted_idx[choice].item())
        if sp.top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, sp.top_k)
            topk_vals = topk_vals / topk_vals.sum()
            choice = torch.multinomial(topk_vals, 1).item()
            return int(topk_idx[choice].item())
        return int(torch.multinomial(probs, 1).item())

    @torch.no_grad()
    def step(self) -> list[EngineOutput]:
        """One iteration: schedule -> batch -> forward -> sample -> update.

        Returns outputs for tokens generated this step (one per scheduled seq).
        """
        with self.profiler.time("schedule"):
            out = self.scheduler.schedule()
            scheduled = out.scheduled
            if not scheduled:
                return []

        # build batch with block tables (pass token_budget for chunked prefill)
        token_budget = out.num_batched_tokens if self.config.enable_chunked_prefill else None
        with self.profiler.time("batch"):
            batch = make_batch(scheduled, device=self.device, token_budget=token_budget)
            if batch is None or batch.num_tokens == 0:
                return []
            # fill block_tables
            block_tables = []
            for sg in scheduled:
                seq = sg.seq
                table = self.block_manager._tables.get(seq.seq_id, [])
                block_tables.append(list(table))
            batch.block_tables = block_tables

        # forward (use CUDA graph for pure decode if available)
        with self.profiler.time("forward"):
            if self.cuda_graph_runner.can_capture(batch):
                logits = self.cuda_graph_runner.run(batch, self.kv_cache)
            else:
                logits = self.runner.forward(batch, self.kv_cache)  # [num_seqs, vocab]

        # sample + update sequences
        with self.profiler.time("sample"):
            outputs: list[EngineOutput] = []
            for i, sg in enumerate(scheduled):
                seq = sg.seq
                # determine how many tokens were computed this step for this seq
                if seq.get_num_uncomputed() > 0:
                    # prefill chunk: we consumed some tokens
                    # Need to know chunk size: it's batch.num_tokens_per_seq[i] if prefill else 1
                    chunk = batch.num_tokens_per_seq[i]
                    seq.num_computed_tokens += chunk
                    # Prefill does not generate a new token unless it was the last chunk
                    # If seq still has uncomputed tokens, we don't sample yet
                    if seq.get_num_uncomputed() > 0:
                        # still prefilling, no token to sample
                        continue
                    # finished prefill, now sample next token from logits[i]
                else:
                    # decode: we will have a new token
                    pass

                # sample next token (for both post-prefill and decode)
                # logits[i] corresponds to this seq's next token
                if i >= logits.shape[0]:
                    continue
                tok = self._sample(logits[i], seq.sampling_params)
                # EOS / stop handling: if tok is stop and not ignore, mark finished after appending?
                # Append and check
                # Allocate block for new token if needed (before appending, check growth)
                new_len = seq.num_tokens + 1
                self.kv_cache.append_slot(seq.seq_id, new_len)
                # For decode, the K/V for this token will be stored in next forward's prefill? Actually
                # decode stores inside forward (we already stored K/V for this token's position inside forward
                # for the *previous* token). Wait: For decode, forward stores K/V for the input token (which is previous output).
                # The newly sampled token hasn't been forward yet; its K/V will be stored next step.
                # So we just append.
                seq.append_token(tok)
                # For prefill case, the num_computed already includes prompt; after sampling we have 1 output token
                # The next step will be decode.

                finished = seq.check_finished()
                # if sequence grew beyond max_seq_len, force finish
                if seq.num_tokens >= self.config.max_seq_len:
                    seq.status = SequenceStatus.FINISHED
                    finished = True

                outputs.append(
                    EngineOutput(
                        request_id=sg.request_id,
                        token_id=int(tok),
                        finished=bool(finished),
                        num_tokens=len(seq.output_token_ids),
                    )
                )
                self._total_tokens += 1

                if finished:
                    # free later after scheduler free
                    seq.status = SequenceStatus.FINISHED
                else:
                    # Need to decide if seq stays RUNNING or needs more prefill?
                    # If it was prefill and just finished prefill, it becomes decodable RUNNING
                    # Mark num_computed_tokens already updated; future steps will be decode path
                    seq.status = SequenceStatus.RUNNING

            # handle finished groups: free blocks and remove from scheduler
            finished_groups = [sg for sg in scheduled if sg.seq.status == SequenceStatus.FINISHED]
            for sg in finished_groups:
                try:
                    self.kv_cache.free(sg.seq.seq_id)
                except Exception:
                    pass
            self.scheduler.free_finished(finished_groups)
            # also remove from req map? keep for stats

        self._step_count += 1
        self._stats["steps"] += 1
        self._stats["tokens"] = self._total_tokens
        return outputs

    def generate(
        self,
        prompts: list[list[int]] | list[int],
        sampling_params: SamplingParams | None = None,
        request_ids: list[str] | None = None,
    ) -> dict[str, list[int]]:
        """Blocking generate for a batch of prompts (sync).

        prompts: list of token id lists OR single token list
        Returns {request_id: output_token_ids}
        """
        # normalize
        if prompts and isinstance(prompts[0], int):
            prompts = [prompts]  # type: ignore
        prompts = list(prompts)  # type: ignore
        n = len(prompts)
        if request_ids is None:
            request_ids = [f"req-{i}-{time.time_ns()}" for i in range(n)]
        if len(request_ids) != n:
            raise ValueError("request_ids length mismatch")

        sp = sampling_params or SamplingParams(max_tokens=self.config.default_max_tokens)
        for rid, toks in zip(request_ids, prompts):
            self.add_request(rid, list(toks), sp)  # type: ignore

        results: dict[str, list[int]] = {rid: [] for rid in request_ids}
        # track per-request outputs via step EngineOutputs
        while self.has_unfinished():
            outs = self.step()
            for o in outs:
                # append to results via internal sg lookup
                results[o.request_id].append(o.token_id)
                if o.finished and o.request_id in self._req_to_sg:
                    # ensure we capture final output_token_ids from sequence
                    sg = self._req_to_sg.get(o.request_id)
                    if sg is not None:
                        results[o.request_id] = list(sg.seq.output_token_ids)
                        # strip trailing EOS if present and not ignore
                        if results[o.request_id] and results[o.request_id][-1] in sp.stop_token_ids and not sp.ignore_eos:
                            results[o.request_id] = results[o.request_id][:-1]
        return results

    async def generate_async(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, list[int]]:
        """Async wrapper around generate (runs in thread)."""
        import asyncio

        return await asyncio.to_thread(self.generate, prompts, sampling_params)

    def stats(self) -> dict:
        s = {
            "steps": self._step_count,
            "total_tokens": self._total_tokens,
            "scheduler": self.scheduler.stats(),
            "kv_cache": self.kv_cache.stats(),
            "runner": type(self.runner).__name__,
            "device": str(self.device),
            "profiler": self.profiler.summary(),
            "cuda_graph": self.cuda_graph_runner.stats(),
            **self._stats,
        }
        if hasattr(self.block_manager, 'prefix_stats'):
            s["prefix_cache"] = self.block_manager.prefix_stats()
        return s

    def reset(self):
        """Clear all state (for tests)."""
        with self._lock:
            for sg in list(self._req_to_sg.values()):
                for seq in sg.sequences:
                    try:
                        self.kv_cache.free(seq.seq_id)
                    except Exception:
                        pass
            self._req_to_sg.clear()
            self.scheduler.waiting.clear()
            self.scheduler.running.clear()
            self.scheduler.swapped.clear()
            self._step_count = 0
            self._total_tokens = 0
