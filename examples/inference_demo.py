#!/usr/bin/env python3
"""
Demo: real inference engine with scheduling, batching, KV cache, memory mgmt.

Runs on CPU (or CUDA if available) with the QwenRunner real Qwen3-0.6B weights.

Shows:
  - Paged KV cache (block tables, gather/store)
  - Continuous batching (prefill + decode mixed)
  - Memory management (block manager stats, OOM handling)
  - Scheduler (FCFS, token budget)

Usage:
  PYTHONPATH=. .venv/bin/python examples/inference_demo.py
  PYTHONPATH=. .venv/bin/python examples/inference_demo.py --qwen  # real weights
  PYTHONPATH=. .venv/bin/python examples/inference_demo.py --bench
"""

import argparse
import time
import sys

def demo_basic():
    from src.inference import InferenceEngine, EngineConfig, SamplingParams
    print("=== Basic single request (QwenRunner, CPU) ===")
    cfg = EngineConfig(device="cpu", num_blocks=32, max_batch_size=4, block_size=16)
    eng = InferenceEngine(cfg, device="cpu", runner="qwen")
    print("KV pool:", eng.kv_cache.stats())
    eng.add_request("demo1", [1, 2, 3, 4, 5], SamplingParams(max_tokens=8, temperature=0.0))
    step = 0
    while eng.has_unfinished():
        outs = eng.step()
        print(f" step {step}: {outs}")
        step += 1
    print(" final tokens:", eng._req_to_sg["demo1"].seq.output_token_ids)
    print(" stats:", eng.stats())
    print()

def demo_batched():
    from src.inference import InferenceEngine, EngineConfig, SamplingParams
    print("=== Batched generate (4 prompts, 1 batch decode = 4x throughput) ===")
    cfg = EngineConfig(device="cpu", num_blocks=64, max_batch_size=4, block_size=16)
    eng = InferenceEngine(cfg, device="cpu", runner="qwen")
    prompts = [[1, 2, 3], [10, 20, 30, 40], [100, 200], [5, 6, 7, 8, 9]]
    t0 = time.time()
    results = eng.generate(prompts, SamplingParams(max_tokens=6), request_ids=["a", "b", "c", "d"])
    dt = time.time() - t0
    for rid, toks in results.items():
        print(f" {rid}: {toks}")
    print(f" done in {dt*1000:.1f}ms, steps={eng.stats()['steps']} (vs 24 sequential steps)")
    print(f" KV free after: {eng.kv_cache.free_blocks}/{eng.kv_cache.config.num_blocks}")
    print(f" profiler:\n{eng.profiler.report()}")
    print()

def demo_continuous():
    from src.inference import InferenceEngine, EngineConfig, SamplingParams
    print("=== Continuous batching (staggered arrivals, mixed prefill+decode) ===")
    cfg = EngineConfig(device="cpu", num_blocks=64, max_batch_size=4, block_size=16)
    eng = InferenceEngine(cfg, device="cpu", runner="qwen")
    eng.add_request("r1", [1, 2, 3], SamplingParams(max_tokens=6))
    eng.add_request("r2", [4, 5, 6], SamplingParams(max_tokens=6))
    for _ in range(2):
        outs = eng.step()
        print(f" step: decode batch {len(outs)} {[o.request_id for o in outs]}")
    print(" -- adding r3/r4 while r1/r2 still decoding (continuous) --")
    eng.add_request("r3", [7, 8, 9], SamplingParams(max_tokens=4))
    eng.add_request("r4", [10, 11, 12], SamplingParams(max_tokens=4))
    step = 2
    while eng.has_unfinished():
        outs = eng.step()
        print(f" step {step}: {[o.request_id for o in outs]} free_blocks={eng.kv_cache.free_blocks} running={eng.scheduler.stats()['running']}")
        step += 1
    print(" all finished, KV leaks?", eng.kv_cache.stats())
    print()

def demo_memory():
    from src.inference import InferenceEngine, EngineConfig, SamplingParams
    print("=== Memory management (tiny 4-block pool, future reservation prevents OOM) ===")
    cfg = EngineConfig(device="cpu", num_blocks=4, max_batch_size=8, block_size=16)
    eng = InferenceEngine(cfg, device="cpu", runner="qwen")
    print(" pool:", eng.kv_cache.stats())
    for i in range(3):
        eng.add_request(f"m{i}", [1]*10, SamplingParams(max_tokens=5))
    pending = eng.scheduler.stats()
    print(f" after enqueue 3 (each 15 tokens -> 1 block but future 2 blocks): waiting={pending['waiting']} free={pending['free_blocks']}")
    # engine will serialize to avoid deadlock
    while eng.has_unfinished():
        outs = eng.step()
        if outs:
            print(f"  step: generated {len(outs)} token(s), free={eng.kv_cache.free_blocks}")
    print(" done, all blocks freed:", eng.kv_cache.free_blocks == eng.kv_cache.config.num_blocks)
    print()

def demo_qwen():
    from src.inference import InferenceEngine, EngineConfig, SamplingParams
    print("=== QwenRunner (real Qwen3-0.6B weights, paged KV, batched) ===")
    try:
        cfg = EngineConfig(model="Qwen/Qwen3-0.6B", device="cpu", num_blocks=32, max_batch_size=2, block_size=16, max_seq_len=128)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        print(f" runner: {type(eng.runner).__name__} loaded={getattr(eng.runner, 'loaded', False)}")
        print(f" KV: {eng.kv_cache.stats()}")
        # tokenize via AutoTokenizer if available
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            prompts = [tok.encode("Hello"), tok.encode("What is AI?")]
            ids = ["hello", "what_is_ai"]
        except Exception:
            prompts = [[1, 2, 3, 4], [10, 20, 30]]
            ids = ["r1", "r2"]
        results = eng.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0), request_ids=ids)
        for rid, out_ids in results.items():
            print(f" {rid}: {out_ids}")
            try:
                print(f"   text: {tok.decode(out_ids)}")
            except Exception:
                pass
        print(" Qwen throughput:", eng.stats())
    except Exception as e:
        print(f" Qwen demo skipped: {e}")
        import traceback; traceback.print_exc()
    print()

def bench():
    from src.inference import InferenceEngine, EngineConfig, SamplingParams
    print("=== Throughput bench (batch 1 vs batch 4) ===")
    import torch
    for bs in [1, 4]:
        cfg = EngineConfig(device="cpu", num_blocks=64, max_batch_size=bs, block_size=16, max_num_batched_tokens=128)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        prompts = [[i]*5 for i in range(4)]
        ids = [f"b{bs}_{i}" for i in range(4)]
        t0 = time.perf_counter()
        eng.generate(prompts, SamplingParams(max_tokens=16), request_ids=ids)
        dt = time.perf_counter() - t0
        toks = 4*16
        print(f" batch_size={bs}: {toks} tokens in {dt*1000:.1f}ms = {toks/dt:.1f} tok/s, steps={eng.stats()['steps']}")
    print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", action="store_true", help="run Qwen real weights demo")
    ap.add_argument("--bench", action="store_true", help="run throughput bench")
    ap.add_argument("--all", action="store_true", help="run all demos")
    args = ap.parse_args()
    if args.all:
        demo_basic()
        demo_batched()
        demo_continuous()
        demo_memory()
        demo_qwen()
        bench()
    elif args.qwen:
        demo_qwen()
    elif args.bench:
        bench()
    else:
        demo_basic()
        demo_batched()
        demo_continuous()
        demo_memory()
        if args.qwen:
            demo_qwen()
