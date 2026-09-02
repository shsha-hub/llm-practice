"""CLI 비교와 Streamlit 앱이 공유하는 생성·중단·정리 로직."""

from __future__ import annotations

import re

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class TaskStoppingCriteria(StoppingCriteria):
    """Task의 출력 형식이 완성되면 EOS를 기다리지 않고 생성을 끝낸다."""

    def __init__(self, tokenizer, prompt_length: int, task: str):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.task = task
        self.fridge_marker_token: int | None = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        new_ids = input_ids[0, self.prompt_length :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        if self.task == "reply":
            return len(re.findall(r"[.!?](?:\s|$)", text)) >= 2

        marker = "소진 재료:"
        if marker in text:
            if self.fridge_marker_token is None:
                self.fridge_marker_token = len(new_ids)
            return len(new_ids) - self.fridge_marker_token >= 18
        return False


def clean_answer(task: str, text: str) -> str:
    """Task 형식 이후에 이어지는 소형 모델의 반복 생성을 제거한다."""
    text = text.strip()
    if task == "reply":
        sentences = re.findall(r".*?[.!?](?=\s|$)", text)
        return " ".join(sentence.strip() for sentence in sentences[:2]) or text

    lines = []
    for line in text.splitlines():
        lines.append(line.rstrip())
        if line.strip().startswith("소진 재료:"):
            break
    return "\n".join(lines).strip()


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, task: str) -> str:
    """선택한 task의 형식에 맞춰 greedy decoding한다."""
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    stopper = TaskStoppingCriteria(tokenizer, encoded["input_ids"].shape[1], task)
    output = model.generate(
        **encoded,
        max_new_tokens=64 if task == "reply" else 120,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        stopping_criteria=StoppingCriteriaList([stopper]),
    )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return clean_answer(task, tokenizer.decode(new_tokens, skip_special_tokens=True))

