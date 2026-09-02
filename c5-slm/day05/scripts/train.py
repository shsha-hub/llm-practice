"""RTX 3050 4GB용 QLoRA 학습 스크립트."""

from __future__ import annotations

import argparse
import gc
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from day05.life_assistant.config import MODEL_ID, TASKS, training_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--smoke", action="store_true", help="1 step만 실행하고 smoke 출력에 저장")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = TASKS[args.task]
    output_dir = task["adapter"].with_name(task["adapter"].name + ("-smoke" if args.smoke else ""))
    set_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다. WSL/Jupyter GPU 연결을 확인하세요.")

    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} / total {total_gib:.2f} GiB / free {free_gib:.2f} GiB")
    if free_gib < 3.2:
        raise RuntimeError("여유 VRAM이 3.2 GiB 미만입니다. 다른 Jupyter/Python 커널을 종료하세요.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = load_dataset("json", data_files=str(task["train"]), split="train")
    val = load_dataset("json", data_files=str(task["val"]), split="train")
    train = train.map(lambda row: training_pair(args.task, row), remove_columns=train.column_names)
    val = val.map(lambda row: training_pair(args.task, row), remove_columns=val.column_names)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    config = SFTConfig(
        output_dir=str(output_dir),
        seed=42,
        num_train_epochs=3,
        max_steps=1 if args.smoke else -1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_length=192,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        optim="paged_adamw_8bit",
        completion_only_loss=True,
        logging_steps=1 if args.smoke else 5,
        eval_strategy="no" if args.smoke else "epoch",
        save_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train,
        eval_dataset=None if args.smoke else val,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"학습 완료: {elapsed:.1f}초 / peak allocated VRAM {peak_gib:.2f} GiB")
    print(f"train_loss: {result.training_loss:.4f}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"어댑터 저장: {output_dir}")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
