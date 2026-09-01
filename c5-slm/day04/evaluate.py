"""골든셋으로 검색 및 재정렬 성능을 평가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from day04.core import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    WasteSearchEngine,
    make_embedder,
    make_reranker,
)


GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "golden_queries.jsonl"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(rankings: list[list[str]], relevant_sets: list[set[str]]) -> dict[str, float]:
    if not rankings:
        raise ValueError("평가 결과가 없습니다.")
    if len(rankings) != len(relevant_sets):
        raise ValueError("검색 결과 수와 정답 수가 다릅니다.")
    hit1 = hit3 = reciprocal_rank = 0.0
    for ranking, relevant in zip(rankings, relevant_sets):
        hit1 += bool(ranking[:1] and ranking[0] in relevant)
        hit3 += bool(set(ranking[:3]) & relevant)
        for rank, rule_id in enumerate(ranking, 1):
            if rule_id in relevant:
                reciprocal_rank += 1 / rank
                break
    count = len(rankings)
    return {
        "Hit@1": round(hit1 / count, 3),
        "Hit@3": round(hit3 / count, 3),
        "MRR": round(reciprocal_rank / count, 3),
    }


def evaluate(engine: WasteSearchEngine, golden: list[dict], reranker=None, k: int = 5):
    rankings = []
    for row in golden:
        response = engine.search(row["query"], k=k, reranker=reranker)
        rankings.append([result.rule.id for result in response.results])
    relevant = [set(row["relevant"]) for row in golden]
    return metrics(rankings, relevant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--with-reranker", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    engine = WasteSearchEngine(
        make_embedder(args.embedding_model, args.device), args.embedding_model
    )
    golden = load_golden()
    print("임베딩 검색:", evaluate(engine, golden))
    if args.with_reranker:
        reranker = make_reranker(args.reranker_model, args.device)
        print("+ 재정렬:", evaluate(engine, golden, reranker=reranker))


if __name__ == "__main__":
    main()
