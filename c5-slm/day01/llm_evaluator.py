"""로컬 causal LM의 seed별 생성 결과를 평가하는 공용 함수."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_TOPIC_KEYWORDS = (
    "점심",
    "식사",
    "밥",
    "음식",
    "메뉴",
    "식당",
    "도시락",
    "배고",
    "맛있",
)


def require_cuda() -> str:
    """강의 환경과 동일하게 CUDA를 필수로 사용한다."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "현재 Python 환경에서 CUDA를 사용할 수 없습니다. "
            "강의 때 사용한 Jupyter 커널/가상환경인지 확인하세요."
        )
    return "cuda"


def load_model(model_id: str = DEFAULT_MODEL_ID):
    """FP16 모델과 토크나이저를 CUDA에 올린다."""
    device = require_cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    ).to(device)
    model.eval()
    return tokenizer, model


def generate_once(
    model,
    tokenizer,
    prompt: str,
    *,
    seed: int,
    max_new_tokens: int = 100,
    device: str = "cuda",
    **generation_options: Any,
) -> dict[str, Any]:
    """한 seed로 답변을 생성하고 속도 정보를 함께 반환한다."""
    if not prompt.strip():
        raise ValueError("prompt는 비어 있을 수 없습니다.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_options,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output.shape[1] - input_length
    answer = tokenizer.decode(
        output[0, input_length:],
        skip_special_tokens=True,
    ).strip()

    return {
        "seed": seed,
        "answer": answer,
        "new_tokens": int(new_tokens),
        "seconds": round(elapsed, 2),
        "tok_per_s": round(new_tokens / elapsed, 1),
    }


_LEADING_MARKER = re.compile(r"^\s*(?:(?:[-*•])|(?:\d+[.)]))?\s*")


def _clean_start(text: str) -> str:
    """목록 번호나 글머리 기호를 제거해 첫 글자를 평가하기 쉽게 만든다."""
    return _LEADING_MARKER.sub("", text).strip()


def _segments(answer: str) -> list[str]:
    """줄바꿈과 주요 구분기호를 기준으로 후보 구절을 나눈다."""
    return [
        _clean_start(part)
        for part in re.split(r"[\n,，;；]+", answer)
        if _clean_start(part)
    ]


def evaluate_acrostic(
    answer: str,
    *,
    syllables: Sequence[str] = ("점", "심"),
    topic_keywords: Sequence[str] = DEFAULT_TOPIC_KEYWORDS,
) -> dict[str, Any]:
    """이행시의 핵심 패턴·엄격한 형식·주제 키워드를 휴리스틱으로 평가한다.

    keyword_hit은 의미 평가가 아니다. 요청 문장 반복이나 우연한 단어 포함도
    통과할 수 있으므로 최종 주제 평가는 사람이 해야 한다.
    """
    if len(syllables) != 2:
        raise ValueError("현재 평가기는 두 글자 이행시만 지원합니다.")

    parts = _segments(answer)
    first_position = next(
        (index for index, part in enumerate(parts) if part.startswith(syllables[0])),
        None,
    )
    second_position = None
    if first_position is not None:
        second_position = next(
            (
                index
                for index, part in enumerate(parts[first_position + 1 :], first_position + 1)
                if part.startswith(syllables[1])
            ),
            None,
        )
    core_pattern_pass = first_position is not None and second_position is not None

    lines = [_clean_start(line) for line in answer.splitlines() if line.strip()]
    strict_format_pass = (
        len(lines) == 2
        and lines[0].startswith(syllables[0])
        and lines[1].startswith(syllables[1])
    )

    content_lines = [
        line
        for line in lines
        if not ("주제" in line and "이행시" in line)
    ]
    content = " ".join(content_lines)
    matched_keywords = [word for word in topic_keywords if word in content]

    return {
        "core_pattern_pass": core_pattern_pass,
        "strict_format_pass": strict_format_pass,
        "keyword_hit": bool(matched_keywords),
        "matched_keywords": ", ".join(matched_keywords),
    }


def run_seed_experiment(
    model,
    tokenizer,
    prompt: str,
    seeds: Iterable[int],
    *,
    generation_options: dict[str, Any],
    topic_keywords: Sequence[str] = DEFAULT_TOPIC_KEYWORDS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """여러 seed를 순차 실행하고 생성 정보와 자동 평가를 합친다."""
    seed_list = list(seeds)
    rows = []
    total = len(seed_list)

    for completed, seed in enumerate(seed_list, start=1):
        generated = generate_once(
            model,
            tokenizer,
            prompt,
            seed=seed,
            **generation_options,
        )
        evaluated = evaluate_acrostic(
            generated["answer"],
            topic_keywords=topic_keywords,
        )
        rows.append({**generated, **evaluated, "manual_topic_pass": False})
        if progress_callback is not None:
            progress_callback(completed, total)

    return rows


def success_rates(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """자동 평가와 사람 평가의 성공률을 백분율로 계산한다."""
    if not rows:
        return {
            "core_pattern_rate": 0.0,
            "strict_format_rate": 0.0,
            "manual_topic_rate": 0.0,
            "overall_rate": 0.0,
        }

    count = len(rows)
    core = sum(bool(row.get("core_pattern_pass")) for row in rows)
    strict = sum(bool(row.get("strict_format_pass")) for row in rows)
    topic = sum(bool(row.get("manual_topic_pass")) for row in rows)
    overall = sum(
        bool(row.get("strict_format_pass")) and bool(row.get("manual_topic_pass"))
        for row in rows
    )
    return {
        "core_pattern_rate": round(core / count * 100, 1),
        "strict_format_rate": round(strict / count * 100, 1),
        "manual_topic_rate": round(topic / count * 100, 1),
        "overall_rate": round(overall / count * 100, 1),
    }
