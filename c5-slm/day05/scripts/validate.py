"""데이터 누락, 중복, 길이를 학습 전에 검사한다."""

from __future__ import annotations

import json

from transformers import AutoTokenizer

from day05.life_assistant.config import MODEL_ID, TASKS, training_pair


EXPECTED = {"train": 40, "val": 8, "test": 5}


def read_rows(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    for task_name, task in TASKS.items():
        seen_inputs: set[str] = set()
        print(f"\n[{task_name}]")
        for split, expected in EXPECTED.items():
            rows = read_rows(task[split])
            assert len(rows) == expected, f"{task_name}/{split}: {len(rows)} != {expected}"
            lengths = []
            for row in rows:
                required = {"answer", "relation", "situation", "intent"} if task_name == "reply" else {
                    "answer", "ingredients", "condition"
                }
                assert required <= row.keys(), f"필드 누락: {row}"
                pair = training_pair(task_name, row)
                key = pair["prompt"]
                assert key not in seen_inputs, f"split 간 중복 입력: {key}"
                seen_inputs.add(key)
                lengths.append(len(tokenizer(pair["prompt"] + pair["completion"], add_special_tokens=False).input_ids))
            print(f"{split}: {len(rows)}개 / token min={min(lengths)}, max={max(lengths)}")
            assert max(lengths) <= 192, f"{task_name}/{split}: 192 token 초과"
    print("\n모든 데이터 검증 통과")


if __name__ == "__main__":
    main()
