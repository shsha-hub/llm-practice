"""Ollama 서버 없이 실행하는 벤치마크 로직 단위 테스트."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ollama_benchmark import (
    ModelInfo,
    OllamaError,
    benchmark_model,
    generate_with_fallback,
    infer_quantization,
    recommend_models,
)


class FakeClient:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.unloaded = []

    def generate(self, model, prompt, **options):
        if model in self.failures:
            raise OllamaError("out of memory")
        return {
            "response": f"{model}: {prompt}",
            "eval_count": 20,
            "eval_duration": 1_000_000_000,
        }

    def vram_gb(self, model):
        return 2.5

    def unload(self, model):
        self.unloaded.append(model)


class BenchmarkTest(unittest.TestCase):
    def test_infer_quantization(self):
        self.assertEqual(infer_quantization("qwen3-q4_k_m:latest"), "Q4_K_M")
        self.assertEqual(infer_quantization("exaone3.5:latest"), "unknown")

    @patch("ollama_benchmark.time.perf_counter", side_effect=[10.0, 12.0])
    def test_benchmark_success(self, _clock):
        client = FakeClient()
        model = ModelInfo("qwen-q4", 2.4, "Q4_K_M")

        result = benchmark_model(client, model, "질문")

        self.assertEqual(result["status"], "성공")
        self.assertEqual(result["seconds"], 2.0)
        self.assertEqual(result["tok_per_s"], 20.0)
        self.assertEqual(client.unloaded, ["qwen-q4"])

    def test_fallback_stops_after_first_success(self):
        client = FakeClient(failures={"large"})
        models = [
            ModelInfo("large", 8.0, "Q8_0"),
            ModelInfo("small", 2.0, "Q2_K"),
            ModelInfo("unused", 1.0, "Q2_K"),
        ]

        result, attempts = generate_with_fallback(client, models, "질문")

        self.assertEqual(result["name"], "small")
        self.assertEqual([row["name"] for row in attempts], ["large", "small"])

    def test_recommendations_ignore_failed_rows(self):
        rows = [
            {"name": "fast", "status": "성공", "tok_per_s": 50, "vram_gb": 4},
            {"name": "small", "status": "성공", "tok_per_s": 20, "vram_gb": 1},
            {"name": "failed", "status": "실패", "tok_per_s": 999, "vram_gb": 0},
        ]

        result = recommend_models(rows)

        self.assertEqual(result["fastest"], "fast")
        self.assertEqual(result["smallest"], "small")


if __name__ == "__main__":
    unittest.main()
