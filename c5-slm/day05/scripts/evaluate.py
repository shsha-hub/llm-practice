"""저장된 비교 결과의 형식 준수율을 간단히 계산한다."""

from __future__ import annotations

import argparse
import json
import re

from day05.life_assistant.config import TASKS


CONDITIONS = ("base", "prompted_base", "adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    return parser.parse_args()


def keyword_check(answer: str, row: dict) -> bool:
    return all(keyword in answer for keyword in row["expected_keywords"])


def reply_checks(answer: str, row: dict) -> dict[str, bool]:
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", answer))
    return {
        "two_or_three_sentences": 2 <= sentence_count <= 3,
        "no_meta_explanation": not any(word in answer for word in ("답변:", "설명:", "예시:")),
        "reasonable_length": len(answer) <= 180,
        "intent_keywords_preserved": keyword_check(answer, row),
    }


def fridge_checks(answer: str, row: dict) -> dict[str, bool]:
    step_count = len(re.findall(r"(?m)^\d+\. ", answer))
    extra_match = re.search(r"(?m)^추가 재료:\s*(.+)$", answer)
    extras = []
    if extra_match and extra_match.group(1).strip() != "없음":
        extras = [item.strip() for item in extra_match.group(1).split(",") if item.strip()]
    return {
        "required_sections": all(
            marker in answer for marker in ("추천:", "추가 재료:", "조리:", "소진 재료:")
        ),
        "three_or_four_steps": 3 <= step_count <= 4,
        "at_most_two_extras": len(extras) <= 2,
        "no_repeated_sections": answer.count("소진 재료:") == 1,
        "input_ingredients_preserved": keyword_check(answer, row),
    }


def main() -> None:
    args = parse_args()
    path = TASKS[args.task]["adapter"].parent / f"comparison-{args.task}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    with TASKS[args.task]["test"].open(encoding="utf-8") as file:
        test_rows = [json.loads(line) for line in file if line.strip()]
    checker = reply_checks if args.task == "reply" else fridge_checks

    for condition in CONDITIONS:
        checked = [checker(row[condition], test_row) for row, test_row in zip(rows, test_rows)]
        total = sum(len(result) for result in checked)
        passed = sum(sum(result.values()) for result in checked)
        details = {
            key: f"{sum(result[key] for result in checked)}/{len(checked)}"
            for key in checked[0]
        }
        print(f"{condition}: {passed}/{total} ({passed / total:.1%}) / {details}")


if __name__ == "__main__":
    main()
