"""Capacity-planner tests: estimator math + probe safety (CPU-safe, no GPU)."""
import unittest


class TestCapacity(unittest.TestCase):
    def test_kv_cache_math(self):
        from runtime.capacity import kv_cache_mb
        # 2*24*2*512*64*2B / MiB = 6.0MB
        self.assertAlmostEqual(kv_cache_mb(), 6.0, places=6)

    def test_session_price_grows_with_genlen(self):
        from runtime.capacity import estimate_session_mb
        short = estimate_session_mb(max_new_tokens=16)["total_mb"]
        long = estimate_session_mb(max_new_tokens=128)["total_mb"]
        self.assertGreater(long, short)
        self.assertGreater(short, 0)

    def test_plan_clamps(self):
        from runtime.capacity import plan_capacity
        p = plan_capacity(4096, 1807, max_new_tokens=48)
        self.assertGreaterEqual(p["max_sessions"], 1)
        self.assertLessEqual(p["max_sessions"], 1000)
        self.assertEqual(p["generation_length"], 48)
        self.assertFalse(p["conservative"])

    def test_plan_unknown_vram_conservative(self):
        from runtime.capacity import plan_capacity
        p = plan_capacity(0, 0)
        self.assertEqual(p["max_sessions"], 1)
        self.assertTrue(p["conservative"])

    def test_longer_genlen_fewer_sessions(self):
        from runtime.capacity import plan_capacity
        a = plan_capacity(4096, 1807, max_new_tokens=16)["max_sessions"]
        b = plan_capacity(4096, 1807, max_new_tokens=256)["max_sessions"]
        self.assertGreaterEqual(a, b)

    def test_probe_never_raises(self):
        from runtime.capacity import probe_vram
        info = probe_vram()
        for k in ("total_mb", "free_mb", "cuda", "source", "name"):
            self.assertIn(k, info)


if __name__ == "__main__":
    unittest.main()
