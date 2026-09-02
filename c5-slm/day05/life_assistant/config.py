"""Day 05 생활 도우미 task별 프롬프트와 데이터 변환."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "K-intelligence/Midm-2.0-Mini-Instruct"

TASKS = {
    "reply": {
        "train": ROOT / "data/reply_train.jsonl",
        "val": ROOT / "data/reply_val.jsonl",
        "test": ROOT / "data/reply_test.jsonl",
        "adapter": ROOT / "outputs/adapter-reply",
        "instruction": (
            "아래 상황에서 상대에게 바로 보낼 한국어 답장만 2~3문장으로 작성하라. "
            "핵심 의사를 분명히 하고 입력된 날짜·금액·조건을 보존하라. "
            "설명, 제목, 따옴표는 쓰지 마라."
        ),
    },
    "fridge": {
        "train": ROOT / "data/fridge_train.jsonl",
        "val": ROOT / "data/fridge_val.jsonl",
        "test": ROOT / "data/fridge_test.jsonl",
        "adapter": ROOT / "outputs/adapter-fridge",
        "instruction": (
            "입력 재료를 우선 소진하는 간단한 한국어 레시피를 작성하라. "
            "추천, 추가 재료, 조리 3~4단계, 소진 재료 순서로만 답하라. "
            "기본 양념 외 추가 핵심 재료는 최대 2개로 제한하라."
        ),
    },
}


def user_text(task: str, row: dict[str, Any]) -> str:
    """원시 JSON 한 행을 모델에 줄 짧은 사용자 입력으로 바꾼다."""
    if task == "reply":
        return (
            f"관계: {row['relation']}\n"
            f"상황: {row['situation']}\n"
            f"의도: {row['intent']}"
        )
    if task == "fridge":
        return f"재료: {row['ingredients']}\n조건: {row['condition']}"
    raise ValueError(f"지원하지 않는 task: {task}")


def training_pair(task: str, row: dict[str, Any]) -> dict[str, str]:
    """Mi:dm의 긴 기본 system prompt를 우회하는 prompt/completion을 만든다."""
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text(task, row)}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return {"prompt": prompt, "completion": row["answer"] + "<|eot_id|>"}


def generation_prompt(task: str, row: dict[str, Any], *, with_instruction: bool) -> str:
    """비교 평가용 생성 프롬프트를 만든다."""
    blocks = ["<|begin_of_text|>"]
    if with_instruction:
        blocks.append(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            + TASKS[task]["instruction"]
            + "<|eot_id|>"
        )
    blocks.append(
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + user_text(task, row)
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return "".join(blocks)
