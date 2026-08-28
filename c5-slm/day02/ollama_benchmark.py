"""Ollama 모델의 자원 사용량과 생성 속도를 비교하는 공용 함수."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Ollama 연결 또는 응답 처리 중 발생한 오류."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_gb: float
    quantization: str


def _gb(byte_count: int | float | None) -> float:
    """Ollama의 byte 값을 십진 GB로 변환한다."""
    return round(float(byte_count or 0) / 1_000_000_000, 2)


def _model_key(name: str) -> str:
    """`model`과 `model:latest`를 같은 모델로 비교한다."""
    return name.removesuffix(":latest")


class OllamaClient:
    """벤치마크에 필요한 Ollama REST API의 작은 래퍼."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _get(self, path: str, *, timeout: float = 10) -> dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}{path}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Ollama GET {path} 실패: {exc}") from exc

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}{path}", json=payload, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Ollama POST {path} 실패: {exc}") from exc
        if data.get("error"):
            raise OllamaError(str(data["error"]))
        return data

    def list_models(self) -> list[ModelInfo]:
        data = self._get("/api/tags")
        models = []
        for item in data.get("models", []):
            details = item.get("details") or {}
            models.append(
                ModelInfo(
                    name=item["name"],
                    size_gb=_gb(item.get("size")),
                    quantization=details.get("quantization_level") or infer_quantization(
                        item["name"]
                    ),
                )
            )
        return sorted(models, key=lambda model: (quantization_bits(model.quantization), model.name))

    def vram_gb(self, model_name: str) -> float:
        for item in self._get("/api/ps").get("models", []):
            if _model_key(item.get("name", "")) == _model_key(model_name):
                return _gb(item.get("size_vram"))
        return 0.0

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        num_predict: int = 120,
        temperature: float = 0.0,
        timeout: float = 180,
    ) -> dict[str, Any]:
        return self._post(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": num_predict,
                    "temperature": temperature,
                },
            },
            timeout=timeout,
        )

    def unload(self, model: str) -> None:
        self._post(
            "/api/generate",
            {"model": model, "keep_alive": 0},
            timeout=30,
        )


_QUANTIZATION_PATTERN = re.compile(r"(?<![A-Z0-9])(Q[2-8](?:_[A-Z0-9]+)*)", re.I)


def infer_quantization(model_name: str) -> str:
    match = _QUANTIZATION_PATTERN.search(model_name)
    return match.group(1).upper() if match else "unknown"


def quantization_bits(quantization: str) -> int:
    match = re.search(r"[2-8]", quantization)
    return int(match.group()) if match else 99


def benchmark_model(
    client: OllamaClient,
    model: ModelInfo,
    prompt: str,
    *,
    num_predict: int = 120,
    temperature: float = 0.0,
    unload_after: bool = True,
) -> dict[str, Any]:
    """모델 하나를 생성하고 시간, 속도, VRAM을 한 행으로 반환한다."""
    if not prompt.strip():
        raise ValueError("prompt는 비어 있을 수 없습니다.")

    started = time.perf_counter()
    try:
        response = client.generate(
            model.name,
            prompt,
            num_predict=num_predict,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - started
        token_count = int(response.get("eval_count") or 0)
        eval_seconds = float(response.get("eval_duration") or 0) / 1_000_000_000
        speed = token_count / eval_seconds if eval_seconds > 0 else token_count / elapsed
        return {
            **asdict(model),
            "status": "성공",
            "seconds": round(elapsed, 2),
            "tokens": token_count,
            "tok_per_s": round(speed, 1),
            "vram_gb": client.vram_gb(model.name),
            "answer": str(response.get("response") or "").strip(),
            "error": "",
        }
    except (OllamaError, requests.RequestException) as exc:
        return {
            **asdict(model),
            "status": "실패",
            "seconds": round(time.perf_counter() - started, 2),
            "tokens": 0,
            "tok_per_s": 0.0,
            "vram_gb": 0.0,
            "answer": "",
            "error": str(exc),
        }
    finally:
        if unload_after:
            try:
                client.unload(model.name)
            except OllamaError:
                pass


def generate_with_fallback(
    client: OllamaClient,
    models: Iterable[ModelInfo],
    prompt: str,
    *,
    num_predict: int = 120,
    temperature: float = 0.0,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """앞 모델부터 시도해 최초 성공 결과와 전체 시도 기록을 반환한다."""
    attempts = []
    for model in models:
        result = benchmark_model(
            client,
            model,
            prompt,
            num_predict=num_predict,
            temperature=temperature,
        )
        attempts.append(result)
        if result["status"] == "성공":
            return result, attempts
    return None, attempts


def recommend_models(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """성공 결과에서 속도·크기·균형 추천을 계산한다.

    균형 점수는 정확도 평가가 아니라 생성 속도와 메모리 효율만 비교한다.
    """
    successful = [row for row in rows if row.get("status") == "성공"]
    if not successful:
        return {}

    fastest = max(successful, key=lambda row: float(row.get("tok_per_s") or 0))
    smallest = min(
        successful,
        key=lambda row: float(row.get("vram_gb") or row.get("size_gb") or 0),
    )
    max_speed = max(float(row.get("tok_per_s") or 0) for row in successful) or 1
    sizes = [float(row.get("vram_gb") or row.get("size_gb") or 0) for row in successful]
    max_size = max(sizes) or 1

    def balance_score(row: dict[str, Any]) -> float:
        speed_score = float(row.get("tok_per_s") or 0) / max_speed
        size_score = 1 - float(row.get("vram_gb") or row.get("size_gb") or 0) / max_size
        return speed_score * 0.6 + size_score * 0.4

    balanced = max(successful, key=balance_score)
    return {
        "fastest": str(fastest["name"]),
        "smallest": str(smallest["name"]),
        "balanced": str(balanced["name"]),
    }
