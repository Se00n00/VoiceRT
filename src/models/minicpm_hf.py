"""MiniCPM5-1B BF16 runner: stock Hugging Face eager, no Triton.

Plain ``transformers`` ``AutoModelForCausalLM`` + ``AutoTokenizer``.
Used as the default until mixed-quant (4-bit bulk + BF16 important layers)
is ready. Matches the :class:`QwenFused` / :class:`MiniCPMFused` contract
so ``LlmModel`` can select it via ``LlmConfig(backend="minicpm")``.

No fused kernels, no GGUF, no custom device code — just ``torch.bfloat16``
on the requested device. ``max_len`` guards prompt+gen the same way the
other runners do so the shared 512/1024 budgeting still holds.
"""
import time

import torch

from src.models.runtime.device import max_allocated_mb
from src.models.runtime.memory import check_budget

__all__ = ["MiniCPMHF", "stop_on_subsequence"]


def stop_on_subsequence(stop_seqs):
    """HF ``StoppingCriteriaList`` halting when input_ids ends with a stop seq.

    ``stop_seqs`` are token-id lists (e.g. ``tok.encode("</function>")``).
    Pure torch — unit-testable without weights. Only batch row 0 is
    checked (terminal turns are B=1).
    """
    from transformers.generation import StoppingCriteria, StoppingCriteriaList

    seqs = [list(s) for s in (stop_seqs or []) if list(s)]
    if not seqs:
        return None

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            tail_len = max(len(s) for s in seqs)
            tail = input_ids[0, -tail_len:].tolist()
            return any(tail[-len(s):] == s for s in seqs)

    return StoppingCriteriaList([_Stop()])


class MiniCPMHF(torch.nn.Module):
    """BF16 eager MiniCPM5-1B via transformers. No Triton."""

    def __init__(self, model="openbmb/MiniCPM5-1B", device="cuda:0",
                 max_len=1024, max_new_tokens=64, **_):
        super().__init__()
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model_id = model
        self.max_len = int(max_len)
        self.max_new_tokens = int(max_new_tokens)
        # keep import local so `import src.models.minicpm` stays light
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # tokenizer is owned by LlmModel, but load here for generate parity
        # (no chat template here — ids come pre-templated from LlmModel.encode).
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        self.tok = tok  # kept for stop-sequence encoding (native tool envelope)
        eos_id = tok.eos_token_id
        if eos_id is None:
            eos_id = tok.convert_tokens_to_ids(tok.eos_token) if tok.eos_token else 1
        self.eos_ids = (int(eos_id),) if eos_id is not None else (1,)
        # model: BF16, let transformers place it (meta -> dispatch if accelerate present)
        try:
            # accelerate available -> device_map="auto" shards if needed
            import accelerate  # noqa: F401
            has_accel = True
        except Exception:
            has_accel = False
        # prefer cuda if requested, else cpu
        if has_accel and device.startswith("cuda"):
            mdl = AutoModelForCausalLM.from_pretrained(
                model, dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=False, low_cpu_mem_usage=True)
        else:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            mdl = AutoModelForCausalLM.from_pretrained(
                model, dtype=dtype, trust_remote_code=False,
                low_cpu_mem_usage=True)
            mdl = mdl.to(device)
        mdl.eval()
        self.model = mdl
        # rough VRAM check: BF16 weights ~2.1GB + KV headroom
        try:
            check_budget(2200.0, budget_mb=4000, what="MiniCPMHF BF16 weights")
        except Exception:
            pass
        print(f"[MiniCPMHF] device={device} model={model} max_len={self.max_len}", flush=True)

    @torch.no_grad()
    def generate(self, ids_batch, max_new_tokens=32, stop_strings=None):
        """Greedy decode. ids_batch: List[List[int]] or [B,T] or flat. Returns dict.

        ``stop_strings`` (e.g. ``["</function>"]``) halts at the native
        tool-call envelope close so the model can't babble past a
        complete call. Chat replies (no envelope) are unaffected.
        """
        if isinstance(ids_batch, torch.Tensor):
            if ids_batch.dim() == 2:
                ids_list = [ids_batch[b].tolist() for b in range(ids_batch.shape[0])]
            else:
                ids_list = [ids_batch.tolist()]
        else:
            if isinstance(ids_batch, list) and ids_batch and isinstance(ids_batch[0], int):
                ids_list = [ids_batch]
            else:
                ids_list = list(ids_batch)
        B = len(ids_list)
        T0 = max(len(x) for x in ids_list) if ids_list else 0
        if T0 + int(max_new_tokens) > int(self.max_len):
            raise ValueError(
                f"prompt {T0} + max_new_tokens {max_new_tokens} exceeds "
                f"max_len {self.max_len}")
        dev = next(self.model.parameters()).device
        # pad to max length for batch
        max_t = max(len(x) for x in ids_list)
        eos = self.eos_ids[0] if self.eos_ids else 1
        inp = torch.full((B, max_t), eos,
                         dtype=torch.long, device=dev)
        attn = torch.zeros(B, max_t, dtype=torch.long, device=dev)
        for b, ids in enumerate(ids_list):
            inp[b, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[b, :len(ids)] = 1
        t0 = time.perf_counter()
        criteria = None
        if stop_strings:
            seqs = [self.tok.encode(s, add_special_tokens=False)
                    for s in stop_strings]
            criteria = stop_on_subsequence(seqs)
        out = self.model.generate(
            input_ids=inp, attention_mask=attn,
            max_new_tokens=int(max_new_tokens), do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos,
            use_cache=True,
            stopping_criteria=criteria)
        # slice off prompt
        gen_ids = []
        for b in range(B):
            # find where prompt ends (attn mask), then strip prompt
            plen = len(ids_list[b])
            # out[b] is prompt + gen; prompt length in out is max_t (padded), not plen
            # but HF generate returns full sequence including padded prompt;
            # easiest: take last `new` tokens beyond max_t
            full = out[b].tolist()
            # locate first pad? simpler: out length - max_t is new tokens
            new_len = len(full) - max_t
            # for non-padded case (all same length), max_t == plen, so this is exact
            # for padded, the first `max_t - plen` tokens are pads — but they were masked
            # so they do not affect gen; we just trim to `new_len` tail and cut eos
            tail = full[-new_len:] if new_len > 0 else []
            if tail and tail[-1] == (self.eos_ids[0] if self.eos_ids else -1):
                tail = tail[:-1]
            gen_ids.append(tail)
        ttft = time.perf_counter() - t0  # includes full gen, good enough until streamed
        total = sum(len(x) for x in gen_ids)
        # decode_tps not meaningful without per-token timing; report avg
        dt = max(time.perf_counter() - t0, 1e-9)
        flat = gen_ids[0] if B == 1 else gen_ids
        return {"ids": flat if B == 1 else gen_ids, "ttft": ttft,
                "decode_tps": total / dt, "vram_mb": max_allocated_mb()}

    def generate_stream(self, ids, max_new_tokens=32, stop_strings=None):
        """Yield ``(tok_id, ttft)`` live as the model decodes.

        Runs HF ``generate`` in a background thread with a tiny
        id-queue streamer, so :meth:`LlmModel.stream` gets true
        per-token callbacks instead of a burst at the end (the old
        behaviour — the TUI only ever saw the finished reply).
        ``stop_strings`` (e.g. ``["</function>"]``) ends a native tool
        call cleanly at the envelope close. Single prompt only
        (terminal turns are B=1).
        """
        import queue
        import threading

        try:
            from transformers import BaseStreamer
        except ImportError:  # transformers>=5: moved out of top-level
            from transformers.generation.streamers import BaseStreamer

        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            if len(ids) > 1:
                raise ValueError("generate_stream supports a single prompt (B=1)")
            ids = ids[0]
        ids = [int(i) for i in ids]
        if len(ids) + int(max_new_tokens) > int(self.max_len):
            raise ValueError(
                f"prompt {len(ids)} + max_new_tokens {max_new_tokens} exceeds "
                f"max_len {self.max_len}")
        dev = next(self.model.parameters()).device
        inp = torch.tensor([ids], dtype=torch.long, device=dev)
        attn = torch.ones_like(inp)
        eos = self.eos_ids[0] if self.eos_ids else 1

        class _IdQueue(BaseStreamer):
            """Forwards generated token *ids* (not text) to a queue.

            Skips the first ``put``: HF ``generate`` pushes the full
            prompt ``input_ids`` before decoding (same as
            ``TextStreamer(skip_prompt=True)``). Without this the prompt
            streams as fake output and poisons the re-decode.
            """

            def __init__(self):
                self.q: queue.Queue = queue.Queue()
                self.is_first = True

            def put(self, value):
                try:
                    if self.is_first:
                        self.is_first = False
                        return
                    for t in value.flatten().tolist():
                        self.q.put(int(t))
                except Exception:
                    pass

            def end(self):
                self.q.put(None)

        qs = _IdQueue()
        err: dict = {}
        criteria = None
        if stop_strings:
            seqs = [self.tok.encode(s, add_special_tokens=False)
                    for s in stop_strings]
            criteria = stop_on_subsequence(seqs)

        def _target():
            try:
                with torch.no_grad():
                    self.model.generate(
                        input_ids=inp, attention_mask=attn,
                        max_new_tokens=int(max_new_tokens), do_sample=False,
                        eos_token_id=eos, pad_token_id=eos,
                        use_cache=True, streamer=qs,
                        stopping_criteria=criteria)
            except Exception as exc:  # noqa: BLE001 - forwarded to consumer
                err["exc"] = exc
            finally:
                qs.end()

        t0 = time.perf_counter()
        ttft: float | None = None
        th = threading.Thread(target=_target, daemon=True)
        th.start()
        try:
            while True:
                item = qs.q.get()
                if item is None:
                    break
                if item == eos:
                    continue  # generate() strips eos too; keep parity
                if ttft is None:
                    ttft = time.perf_counter() - t0
                yield int(item), float(ttft)
        finally:
            th.join(timeout=60)
        if "exc" in err:
            raise err["exc"]
