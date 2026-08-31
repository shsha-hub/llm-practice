import unittest

from capacity import (
    GIB,
    MIB,
    ModelArchitecture,
    available_kv_bytes,
    estimate_max_users,
    kv_bytes_per_token,
    kv_cache_bytes,
)


class CapacityTest(unittest.TestCase):
    def setUp(self):
        self.qwen = ModelArchitecture(28, 16, 8, 128)

    def test_qwen_fp16_kv_cache(self):
        self.assertEqual(kv_bytes_per_token(self.qwen), 114_688)
        self.assertEqual(kv_cache_bytes(self.qwen, 4096) / MIB, 448)

    def test_mha_uses_twice_the_gqa_memory(self):
        gqa = kv_cache_bytes(self.qwen, 4096)
        mha = kv_cache_bytes(self.qwen, 4096, use_mha=True)
        self.assertEqual(mha, gqa * 2)

    def test_estimated_users_only_use_available_memory(self):
        available = available_kv_bytes(24, 2, memory_utilization=0.9)
        self.assertAlmostEqual(available / GIB, 19.6)
        self.assertEqual(
            estimate_max_users(self.qwen, 4096, 24, 2, memory_utilization=0.9),
            44,
        )

    def test_invalid_architecture(self):
        with self.assertRaises(ValueError):
            ModelArchitecture(28, 8, 16, 128)


if __name__ == "__main__":
    unittest.main()

