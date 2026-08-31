"""OpenAI 호환 LLM 서버를 위한 작은 비동기 부하테스트 도구."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Sequence

from openai import AsyncOpenAI


DEFAULT_PROMPTS = (
    "파이썬으로 리스트를 뒤집는 방법을 한 문장으로 설명해 줘.",
    "재귀 함수가 무엇인지 초보자에게 한 문장으로 설명해 줘.",
    "HTTP와 HTTPS의 차이를 한 문장으로 설명해 줘.",
    "머신러닝과 딥러닝의 차이를 한 문장으로 설명해 줘.",
    "REST API가 무엇인지 한 문장으로 설명해 줘.",
    "가을을 주제로 짧은 문장 하나를 써 줘.",
    "함수형 프로그래밍의 핵심을 한 문장으로 설명해 줘.",
    "대한민국의 수도와 특징을 한 문장으로 설명해 줘.",
)


@dataclass(frozen=True)
class RequestResult:
    latency_s: float
    completion_tokens: int
    success: bool
    error: str = ""


@dataclass(frozen=True)
class LoadResult:
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    wall_time_s: float
    p50_latency_s: float
    p95_latency_s: float
    max_latency_s: float
    requests_per_s: float
    tokens_per_s: float
    completion_tokens: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["errors"] = list(self.errors)
        return data


def percentile(values: Sequence[float], percent: float) -> float:
    """선형 보간으로 percentile을 계산한다."""
    if not values:
        return 0.0
    if not 0 <= percent <= 100:
        raise ValueError("percent는 0에서 100 사이여야 합니다.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def request_once(
    client: Any,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> RequestResult:
    """요청 하나의 end-to-end latency와 생성 토큰 수를 측정한다."""
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout_s,
        )
        latency = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return RequestResult(latency, tokens, True)
    except Exception as exc:  # 서버·SDK·timeout 오류를 요청 결과로 보존한다.
        return RequestResult(
            time.perf_counter() - started,
            0,
            False,
            f"{type(exc).__name__}: {exc}",
        )


async def run_load(
    client: Any,
    model: str,
    prompts: Sequence[str],
    *,
    concurrency: int,
    total_requests: int,
    max_tokens: int = 128,
    temperature: float = 0.3,
    timeout_s: float = 120,
    request_fn: Callable[..., Awaitable[RequestResult]] = request_once,
) -> LoadResult:
    """고정된 요청 수를 semaphore로 제한하며 부하를 발생시킨다."""
    if concurrency <= 0 or total_requests <= 0:
        raise ValueError("concurrency와 total_requests는 0보다 커야 합니다.")
    if not prompts or not all(prompt.strip() for prompt in prompts):
        raise ValueError("비어 있지 않은 prompt가 하나 이상 필요합니다.")

    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(index: int) -> RequestResult:
        async with semaphore:
            return await request_fn(
                client,
                model,
                prompts[index % len(prompts)],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )

    started = time.perf_counter()
    results = await asyncio.gather(
        *(limited_request(index) for index in range(total_requests))
    )
    wall_time = time.perf_counter() - started
    successful = [result for result in results if result.success]
    latencies = [result.latency_s for result in successful]
    tokens = sum(result.completion_tokens for result in successful)
    errors = tuple(dict.fromkeys(result.error for result in results if result.error))

    return LoadResult(
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=len(successful),
        failed_requests=total_requests - len(successful),
        wall_time_s=wall_time,
        p50_latency_s=percentile(latencies, 50),
        p95_latency_s=percentile(latencies, 95),
        max_latency_s=max(latencies, default=0.0),
        requests_per_s=len(successful) / wall_time if wall_time else 0.0,
        tokens_per_s=tokens / wall_time if wall_time else 0.0,
        completion_tokens=tokens,
        errors=errors,
    )


async def benchmark_levels(
    base_url: str,
    api_key: str,
    model: str,
    prompts: Sequence[str],
    concurrency_levels: Sequence[int],
    *,
    total_requests: int,
    max_tokens: int = 128,
    temperature: float = 0.3,
    timeout_s: float = 120,
) -> list[LoadResult]:
    """워밍업 후 각 동시성 조건을 순서대로 측정한다."""
    if not concurrency_levels:
        raise ValueError("동시성 조건이 하나 이상 필요합니다.")
    client = AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "not-needed",
        timeout=timeout_s,
        max_retries=0,
    )
    try:
        warmup = await request_once(
            client,
            model,
            prompts[0],
            max_tokens=min(max_tokens, 32),
            temperature=temperature,
            timeout_s=timeout_s,
        )
        if not warmup.success:
            raise RuntimeError(f"워밍업 요청 실패: {warmup.error}")

        output = []
        for concurrency in concurrency_levels:
            output.append(
                await run_load(
                    client,
                    model,
                    prompts,
                    concurrency=concurrency,
                    total_requests=max(total_requests, concurrency),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
            )
        return output
    finally:
        await client.close()

