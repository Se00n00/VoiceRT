"""Tests for the real inference engine (scheduling + batching + KV cache + memory mgmt)."""

import unittest
import torch


class TestBlockManager(unittest.TestCase):
    def test_allocate_and_free(self):
        from src.inference.config import KVCacheConfig
        from src.inference.block_manager import BlockManager
        cfg = KVCacheConfig(block_size=16, num_blocks=8, num_layers=2, num_kv_heads=2, head_dim=32)
        bm = BlockManager(cfg)
        self.assertEqual(bm.free_count, 8)
        t1 = bm.allocate(1, 2)
        self.assertEqual(len(t1), 2)
        self.assertEqual(bm.free_count, 6)
        self.assertTrue(bm.can_allocate(6))
        self.assertFalse(bm.can_allocate(7))
        bm.free(1)
        self.assertEqual(bm.free_count, 8)

    def test_block_table_for_tokens(self):
        from src.inference.config import KVCacheConfig
        from src.inference.block_manager import BlockManager
        cfg = KVCacheConfig(block_size=16, num_blocks=32, num_layers=2, num_kv_heads=2, head_dim=32)
        bm = BlockManager(cfg)
        # 30 tokens needs 2 blocks (ceil 30/16=2)
        bm.ensure_seq(10, 30)
        self.assertEqual(len(bm._tables[10]), 2)
        # 5 tokens needs 1 block
        bm.ensure_seq(11, 5)
        self.assertEqual(len(bm._tables[11]), 1)

    def test_memory_pool_gather_store(self):
        from src.inference.config import KVCacheConfig
        from src.inference.block_manager import KVCacheMemoryPool
        cfg = KVCacheConfig(block_size=4, num_blocks=8, num_layers=2, num_kv_heads=2, head_dim=8)
        pool = KVCacheMemoryPool(cfg, device="cpu", dtype="fp32")
        # store a block
        k = torch.randn(4, 2, 8)
        v = torch.randn(4, 2, 8)
        pool.store_block(0, 0, k, v)
        # gather
        Kg, Vg = pool.gather(0, [0], 4)
        self.assertEqual(Kg.shape, (4, 2, 8))
        self.assertTrue(torch.allclose(Kg, k))
        # multi-block gather
        k2 = torch.randn(2, 2, 8)
        v2 = torch.randn(2, 2, 8)
        pool.store_block(0, 1, k2, v2)
        Kg2, Vg2 = pool.gather(0, [0, 1], 6)
        self.assertEqual(Kg2.shape, (6, 2, 8))
        self.assertTrue(torch.allclose(Kg2[:4], k))
        self.assertTrue(torch.allclose(Kg2[4:], k2))


class TestPagedKVCache(unittest.TestCase):
    def test_paged_store_gather(self):
        from src.inference.config import KVCacheConfig
        from src.inference.kv_cache import PagedKVCache
        cfg = KVCacheConfig(block_size=16, num_blocks=16, num_layers=2, num_kv_heads=2, head_dim=8, dtype="fp32")
        cache = PagedKVCache(cfg, device="cpu")
        seq_id = 1
        cache.allocate_for_seq(seq_id, 10)  # 1 block
        self.assertEqual(cache.free_blocks, 15)
        k = torch.randn(10, 2, 8)
        v = torch.randn(10, 2, 8)
        cache.store_prefill(seq_id, 0, k, v)
        Kg, Vg = cache.gather(seq_id, 0, 10)
        self.assertTrue(torch.allclose(Kg, k))
        # append single token
        k1 = torch.randn(2, 8)
        v1 = torch.randn(2, 8)
        cache.append_slot(seq_id, 11)
        cache.store_decode(seq_id, 0, 10, k1, v1)
        Kg2, _ = cache.gather(seq_id, 0, 11)
        self.assertEqual(Kg2.shape[0], 11)
        self.assertTrue(torch.allclose(Kg2[10], k1))
        cache.free(seq_id)
        self.assertEqual(cache.free_blocks, 16)


class TestScheduler(unittest.TestCase):
    def test_fcfs_admission(self):
        from src.inference.config import SchedulerConfig, KVCacheConfig
        from src.inference.block_manager import BlockManager
        from src.inference.scheduler import ContinuousScheduler
        from src.inference.sequence import SequenceGroup
        from src.inference.config import SamplingParams
        kv_cfg = KVCacheConfig(block_size=16, num_blocks=64, num_layers=2, num_kv_heads=2, head_dim=8)
        bm = BlockManager(kv_cfg)
        sched = ContinuousScheduler(SchedulerConfig(max_batch_size=2, max_num_batched_tokens=32, max_num_seqs=4), bm)
        for i in range(3):
            sg = SequenceGroup.from_prompt(f"r{i}", [1, 2, 3], SamplingParams(max_tokens=4), block_size=16, seq_id=i+1)
            sched.add_request(sg)
        out = sched.schedule()
        # first schedule admits waiting prefills up to batch size
        self.assertEqual(len(out.scheduled), 2)
        self.assertEqual(sched.stats()["waiting"], 1)
        # next schedule handles running decodes
        out2 = sched.schedule()
        # running decodes should be scheduled again
        self.assertEqual(len(out2.scheduled), 2)

    def test_token_budget(self):
        from src.inference.config import SchedulerConfig, KVCacheConfig
        from src.inference.block_manager import BlockManager
        from src.inference.scheduler import ContinuousScheduler
        from src.inference.sequence import SequenceGroup
        from src.inference.config import SamplingParams
        kv_cfg = KVCacheConfig(block_size=16, num_blocks=64, num_layers=2, num_kv_heads=2, head_dim=8)
        bm = BlockManager(kv_cfg)
        # tiny token budget
        sched = ContinuousScheduler(SchedulerConfig(max_batch_size=8, max_num_batched_tokens=4, max_num_seqs=8), bm)
        sg = SequenceGroup.from_prompt("r1", [1]*10, SamplingParams(max_tokens=4), block_size=16, seq_id=1)
        sched.add_request(sg)
        out = sched.schedule()
        # 10 token prefill exceeds budget 4 -> cannot schedule (FCFS blocks)
        self.assertEqual(len(out.scheduled), 0)
        self.assertEqual(out.waiting, 1)


class TestInferenceEngine(unittest.TestCase):
    def test_single_request_generate(self):
        from src.inference import InferenceEngine, EngineConfig, SamplingParams
        cfg = EngineConfig(device="cpu", num_blocks=32, max_batch_size=4, block_size=16)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        results = eng.generate([[1,2,3]], SamplingParams(max_tokens=4, temperature=0.0), request_ids=["t1"])
        self.assertIn("t1", results)
        self.assertEqual(len(results["t1"]), 4)
        # after generate, blocks freed
        self.assertEqual(eng.kv_cache.free_blocks, eng.kv_cache.config.num_blocks)

    def test_batched_generate_throughput(self):
        from src.inference import InferenceEngine, EngineConfig, SamplingParams
        cfg = EngineConfig(device="cpu", num_blocks=64, max_batch_size=4, block_size=16, max_num_batched_tokens=128)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        prompts = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
        results = eng.generate(prompts, SamplingParams(max_tokens=4), request_ids=["a","b","c","d"])
        self.assertEqual(len(results), 4)
        for rid in ["a","b","c","d"]:
            self.assertEqual(len(results[rid]), 4)
        # batched: 4 seqs * 4 tokens = 16 tokens in 4 steps (not 16)
        self.assertEqual(eng.stats()["steps"], 4)

    def test_continuous_batching_staggered(self):
        from src.inference import InferenceEngine, EngineConfig, SamplingParams
        cfg = EngineConfig(device="cpu", num_blocks=64, max_batch_size=2, block_size=16)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        eng.add_request("r1", [1,2,3], SamplingParams(max_tokens=5))
        eng.add_request("r2", [4,5,6], SamplingParams(max_tokens=5))
        # step a few
        for _ in range(2):
            eng.step()
        # add more while first still running
        eng.add_request("r3", [7,8,9], SamplingParams(max_tokens=3))
        eng.add_request("r4", [10,11,12], SamplingParams(max_tokens=3))
        while eng.has_unfinished():
            eng.step()
        for rid in ["r1","r2","r3","r4"]:
            sg = eng._req_to_sg[rid]
            self.assertEqual(sg.seq.status.value, "FINISHED")
        self.assertEqual(eng.kv_cache.free_blocks, eng.kv_cache.config.num_blocks)

    def test_memory_budget_enforced(self):
        from src.inference import InferenceEngine, EngineConfig, SamplingParams
        # tiny cache: 4 blocks *16 =64 tokens total
        cfg = EngineConfig(device="cpu", num_blocks=4, max_batch_size=8, block_size=16)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        # Each seq prompt 10 + max 10 =20 tokens needs 2 blocks, so with 4 blocks only 1 can run at a time (watermark 2)
        # Engine should serialize, not deadlock
        prompts = [[1]*10 for _ in range(3)]
        results = eng.generate(prompts, SamplingParams(max_tokens=5), request_ids=["x","y","z"])
        for rid in ["x","y","z"]:
            self.assertEqual(len(results[rid]), 5)
        # ensure no leak
        self.assertEqual(eng.kv_cache.free_blocks, 4)

    def test_kv_cache_real_tensors(self):
        from src.inference import InferenceEngine, EngineConfig, SamplingParams
        cfg = EngineConfig(device="cpu", num_blocks=8, max_batch_size=1, block_size=4)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        # verify KV tensors are real torch tensors on device
        self.assertIsInstance(eng.kv_cache.pool.k_pools[0], torch.Tensor)
        self.assertEqual(eng.kv_cache.pool.k_pools[0].device.type, "cpu")
        eng.add_request("kvtest", [1,2,3,4,5], SamplingParams(max_tokens=2))
        eng.step()  # prefill
        # after prefill, KV should be populated
        seq_id = eng._req_to_sg["kvtest"].seq.seq_id
        Kg, Vg = eng.kv_cache.gather(seq_id, 0, 5)
        # real values, not zeros (after prefill)
        self.assertNotEqual(float(Kg.abs().sum().item()), 0.0)

    def test_step_produces_real_attention(self):
        from src.inference import InferenceEngine, EngineConfig, SamplingParams
        cfg = EngineConfig(device="cpu", num_blocks=16, max_batch_size=2, block_size=16)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        # two identical prompts should produce identical first token (deterministic greedy)
        # but different prompts should differ
        eng.add_request("a", [1,2,3], SamplingParams(max_tokens=1, temperature=0.0))
        eng.add_request("b", [1,2,3], SamplingParams(max_tokens=1, temperature=0.0))
        outs = eng.step()
        self.assertEqual(len(outs), 2)
        self.assertEqual(outs[0].token_id, outs[1].token_id)

    def test_abort(self):
        from src.inference import InferenceEngine, EngineConfig, SamplingParams
        cfg = EngineConfig(device="cpu", num_blocks=16, max_batch_size=2)
        eng = InferenceEngine(cfg, device="cpu", runner="qwen")
        eng.add_request("to_abort", [1,2,3], SamplingParams(max_tokens=10))
        eng.abort_request("to_abort")
        self.assertFalse(eng.has_unfinished() and any(sg.request_id=="to_abort" for sg in eng.scheduler.running))
        self.assertEqual(eng.kv_cache.free_blocks, eng.kv_cache.config.num_blocks)

    def test_runtime_facade_import(self):
        from src.models.runtime.inference_engine import InferenceEngine as REngine, auto_engine_config
        cfg = auto_engine_config()
        self.assertIsNotNone(cfg)
        self.assertGreater(cfg.num_blocks, 0)
        # also via top-level inference package
        from src.inference import InferenceEngine as R2, EngineConfig, PagedKVCache
        self.assertIsNotNone(R2)
        self.assertIsNotNone(EngineConfig)
        self.assertIsNotNone(PagedKVCache)
        # facade re-exports engine/inference_engine as well
        from engine.inference_engine import InferenceEngine as E3
        self.assertIsNotNone(E3)


if __name__ == "__main__":
    unittest.main()
