import asyncio
import unittest
from types import SimpleNamespace

from benchmark import percentile, run_load


class FakeCompletions:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    async def create(self, **kwargs):
        self.calls += 1
        call_number = self.calls
        await asyncio.sleep(0.002)
        if call_number == self.fail_on_call:
            raise RuntimeError("fake failure")
        return SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=10),
            choices=[],
        )


class FakeClient:
    def __init__(self, fail_on_call=None):
        self.chat = SimpleNamespace(completions=FakeCompletions(fail_on_call))


class BenchmarkTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)

    def test_load_result_counts_tokens_and_failures(self):
        result = asyncio.run(
            run_load(
                FakeClient(fail_on_call=2),
                "fake-model",
                ["hello"],
                concurrency=2,
                total_requests=4,
                timeout_s=1,
            )
        )
        self.assertEqual(result.successful_requests, 3)
        self.assertEqual(result.failed_requests, 1)
        self.assertEqual(result.completion_tokens, 30)
        self.assertGreater(result.tokens_per_s, 0)
        self.assertIn("fake failure", result.errors[0])

    def test_rejects_empty_prompts(self):
        with self.assertRaises(ValueError):
            asyncio.run(
                run_load(
                    FakeClient(),
                    "fake-model",
                    [],
                    concurrency=1,
                    total_requests=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
