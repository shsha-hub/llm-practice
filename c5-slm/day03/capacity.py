"""LLM KV cache 메모리와 동시 사용자 수를 계산한다."""

from __future__ import annotations

import math
from dataclasses import dataclass


MIB = 1024**2
GIB = 1024**3

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "fp8": 1,
}


@dataclass(frozen=True)
class ModelArchitecture:
    """KV cache 계산에 필요한 모델 구조 정보."""

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_layers", self.num_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name}은 0보다 커야 합니다.")
        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError("KV head 수는 attention head 수보다 클 수 없습니다.")


def dtype_bytes(dtype: str) -> int:
    """dtype 한 원소의 byte 수를 반환한다."""
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        choices = ", ".join(DTYPE_BYTES)
        raise ValueError(f"지원하지 않는 dtype입니다: {dtype} ({choices})") from exc


def kv_bytes_per_token(
    architecture: ModelArchitecture,
    *,
    dtype: str = "float16",
    use_mha: bool = False,
) -> int:
    """토큰 하나의 전체 layer KV cache 크기를 byte로 계산한다.

    GQA 모델의 실제 계산에는 ``num_kv_heads``를 사용한다. ``use_mha``를
    켜면 비교를 위해 Q head 수만큼 K/V head가 있다고 가정한다.
    """
    heads = (
        architecture.num_attention_heads if use_mha else architecture.num_kv_heads
    )
    return (
        architecture.num_layers
        * heads
        * architecture.head_dim
        * 2  # K와 V
        * dtype_bytes(dtype)
    )


def kv_cache_bytes(
    architecture: ModelArchitecture,
    context_length: int,
    concurrent_users: int = 1,
    *,
    dtype: str = "float16",
    use_mha: bool = False,
) -> int:
    """주어진 컨텍스트와 동시 사용자 수의 KV cache byte를 반환한다."""
    if context_length <= 0:
        raise ValueError("context_length는 0보다 커야 합니다.")
    if concurrent_users <= 0:
        raise ValueError("concurrent_users는 0보다 커야 합니다.")
    return (
        kv_bytes_per_token(architecture, dtype=dtype, use_mha=use_mha)
        * context_length
        * concurrent_users
    )


def available_kv_bytes(
    gpu_memory_gib: float,
    model_memory_gib: float,
    *,
    memory_utilization: float = 0.9,
) -> int:
    """GPU 메모리 중 모델을 제외하고 KV cache에 쓸 수 있는 양을 구한다."""
    if gpu_memory_gib <= 0:
        raise ValueError("gpu_memory_gib는 0보다 커야 합니다.")
    if model_memory_gib < 0:
        raise ValueError("model_memory_gib는 음수일 수 없습니다.")
    if not 0 < memory_utilization <= 1:
        raise ValueError("memory_utilization은 0 초과 1 이하여야 합니다.")
    available_gib = gpu_memory_gib * memory_utilization - model_memory_gib
    return max(0, int(available_gib * GIB))


def estimate_max_users(
    architecture: ModelArchitecture,
    context_length: int,
    gpu_memory_gib: float,
    model_memory_gib: float,
    *,
    dtype: str = "float16",
    memory_utilization: float = 0.9,
) -> int:
    """KV cache 메모리만을 기준으로 한 이론상 최대 동시 사용자 수."""
    per_user = kv_cache_bytes(
        architecture,
        context_length,
        dtype=dtype,
    )
    return math.floor(
        available_kv_bytes(
            gpu_memory_gib,
            model_memory_gib,
            memory_utilization=memory_utilization,
        )
        / per_user
    )

