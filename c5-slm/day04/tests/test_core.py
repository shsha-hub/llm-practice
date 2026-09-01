import tempfile
import unittest
from pathlib import Path

import numpy as np

from day04.evaluate import metrics
from day04.core import (
    WasteRule,
    WasteSearchEngine,
    fallback_answer,
    load_rules,
    load_schedule,
    neighborhood_names,
    schedule_guidance,
)


def rule(rule_id, item, aliases):
    return WasteRule(
        id=rule_id,
        jurisdiction="부산광역시 부산진구",
        item=item,
        aliases=aliases,
        category="테스트 분류",
        schedule_group="recyclable_a",
        instructions="안전하게 배출합니다.",
        cautions="실제 안내를 확인합니다.",
        source_title="테스트 출처",
        source_url="https://example.com",
        published_at="2026-08-01",
        verified_at="2026-09-01",
    )


RULES = [
    rule("glass", "깨진 유리", ["유리 조각"]),
    rule("battery", "폐건전지", ["AA 배터리"]),
]


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def _one(self, text):
        if "유리" in text:
            return np.array([1.0, 0.0, 0.0])
        if "건전지" in text or "배터리" in text:
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])

    def encode(self, texts, **kwargs):
        self.calls += 1
        if isinstance(texts, str):
            return self._one(texts)
        return np.vstack([self._one(text) for text in texts])


class ReverseReranker:
    def predict(self, pairs):
        return [-2.0, 0.0]


class WasteCoreTest(unittest.TestCase):
    def make_engine(self, directory, embedder=None, rules=RULES):
        return WasteSearchEngine(
            embedder or FakeEmbedder(),
            "fake/model",
            rules=rules,
            index_dir=Path(directory),
        )

    def test_semantic_search_finds_matching_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory)
            result = engine.semantic_search("깨진 유리컵", k=1)[0]
            self.assertEqual(result.rule.id, "glass")
            self.assertAlmostEqual(result.semantic_score, 1.0)

    def test_low_confidence_refuses_to_guess(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory)
            response = engine.search("낡은 운동화", confidence_threshold=0.5)
            self.assertFalse(response.confident)
            self.assertIn("확신할 수 없습니다", fallback_answer(response))

    def test_reranker_can_move_zero_score_above_negative_score(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory)
            response = engine.search("유리", k=2, reranker=ReverseReranker())
            self.assertEqual(response.results[0].rule.id, "battery")
            self.assertEqual(response.results[0].rerank_score, 0.0)

    def test_cache_reused_and_corpus_change_triggers_reindex(self):
        with tempfile.TemporaryDirectory() as directory:
            first_embedder = FakeEmbedder()
            first = self.make_engine(directory, first_embedder)
            self.assertFalse(first.loaded_from_cache)
            self.assertEqual(first_embedder.calls, 1)

            second_embedder = FakeEmbedder()
            second = self.make_engine(directory, second_embedder)
            self.assertTrue(second.loaded_from_cache)
            self.assertEqual(second_embedder.calls, 0)

            changed_embedder = FakeEmbedder()
            changed_rules = RULES + [rule("third", "새 규칙", ["추가 품목"])]
            changed = self.make_engine(directory, changed_embedder, changed_rules)
            self.assertFalse(changed.loaded_from_cache)
            self.assertEqual(changed_embedder.calls, 1)

    def test_metrics(self):
        result = metrics(
            [["a", "b", "c"], ["x", "y", "z"]],
            [{"a"}, {"y"}],
        )
        self.assertEqual(result, {"Hit@1": 0.5, "Hit@3": 1.0, "MRR": 0.75})

    def test_busanjin_zones_produce_different_recycling_days(self):
        schedule = load_schedule()
        first_zone = schedule_guidance(RULES[0], "부전1동", schedule)
        second_zone = schedule_guidance(RULES[0], "전포1동", schedule)
        self.assertEqual(first_zone.days, ["월요일"])
        self.assertEqual(second_zone.days, ["화요일"])
        self.assertNotEqual(first_zone.zone, second_zone.zone)

    def test_every_busanjin_neighborhood_appears_once(self):
        schedule = load_schedule()
        names = neighborhood_names(schedule)
        self.assertEqual(len(names), 20)
        self.assertEqual(len(names), len(set(names)))

    def test_unknown_schedule_group_requires_confirmation(self):
        unknown = WasteRule(
            **{**RULES[0].__dict__, "id": "unknown", "schedule_group": "verify"}
        )
        guidance = schedule_guidance(unknown, "부전1동")
        self.assertFalse(guidance.known)
        self.assertIn("확인", guidance.method)

    def test_project_data_has_unique_ids_and_valid_golden_references(self):
        project_dir = Path(__file__).resolve().parents[1]
        rules = load_rules(project_dir / "data" / "waste_rules.jsonl")
        rule_ids = {item.id for item in rules}
        self.assertEqual(len(rules), 20)
        self.assertTrue(all(item.source_url.startswith("https://") for item in rules))

        import json

        golden = [
            json.loads(line)
            for line in (project_dir / "data" / "golden_queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(golden), 15)
        for row in golden:
            self.assertTrue(set(row["relevant"]) <= rule_ids)


if __name__ == "__main__":
    unittest.main()
