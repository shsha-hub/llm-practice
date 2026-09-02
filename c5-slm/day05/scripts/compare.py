"""미학습 테스트셋에서 base / prompted base / adapter를 비교한다."""

from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from day05.life_assistant.config import MODEL_ID, TASKS, generation_prompt, user_text
from day05.life_assistant.inference import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def load_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    args = parse_args()
    task = TASKS[args.task]
    if not task["adapter"].is_dir():
        raise FileNotFoundError(f"어댑터가 없습니다: {task['adapter']}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, str(task["adapter"]))
    rows = load_jsonl(task["test"])[: args.limit]

    results = []
    for index, row in enumerate(rows, 1):
        plain_prompt = generation_prompt(args.task, row, with_instruction=False)
        system_prompt = generation_prompt(args.task, row, with_instruction=True)
        with model.disable_adapter():
            base_answer = generate(model, tokenizer, plain_prompt, args.task)
            prompted_answer = generate(model, tokenizer, system_prompt, args.task)
        adapter_answer = generate(model, tokenizer, plain_prompt, args.task)
        item = {
            "index": index,
            "input": user_text(args.task, row),
            "base": base_answer,
            "prompted_base": prompted_answer,
            "adapter": adapter_answer,
        }
        results.append(item)
        print(f"\n===== TEST {index} =====", flush=True)
        print("[BASE]", base_answer, sep="\n", flush=True)
        print("[PROMPTED BASE]", prompted_answer, sep="\n", flush=True)
        print("[ADAPTER]", adapter_answer, sep="\n", flush=True)

    output = task["adapter"].parent / f"comparison-{args.task}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n비교 결과 저장: {output}")


if __name__ == "__main__":
    main()

