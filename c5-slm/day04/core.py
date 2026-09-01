"""생활폐기물 분리배출 도우미의 검색·재정렬·RAG 코어."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Protocol, Sequence

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = BASE_DIR / "data" / "waste_rules.jsonl"
DEFAULT_SCHEDULE_PATH = BASE_DIR / "data" / "busanjin_schedule.json"
DEFAULT_INDEX_DIR = BASE_DIR / "indexes"
DEFAULT_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Embedder(Protocol):
    def encode(self, texts: str | list[str], **kwargs: object) -> np.ndarray: ...


class PairReranker(Protocol):
    def predict(self, pairs: list[list[str]]) -> Sequence[float]: ...


@dataclass(frozen=True)
class WasteRule:
    id: str
    jurisdiction: str
    item: str
    aliases: list[str]
    category: str
    schedule_group: str
    instructions: str
    cautions: str
    source_title: str
    source_url: str
    published_at: str
    verified_at: str

    @property
    def search_text(self) -> str:
        aliases = ", ".join(self.aliases)
        return (
            f"품목: {self.item}. 다른 표현: {aliases}. 분류: {self.category}. "
            f"배출 방법: {self.instructions} 주의: {self.cautions}"
        )


@dataclass(frozen=True)
class SearchResult:
    rule: WasteRule
    semantic_score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[SearchResult]
    confident: bool
    confidence_threshold: float


@dataclass(frozen=True)
class ScheduleGuidance:
    neighborhood: str
    zone: str
    days: list[str]
    time: str
    place: str
    method: str
    known: bool
    note: str
    contact: str
    source_title: str
    source_url: str
    effective_from: str
    published_at: str
    verified_at: str


def load_rules(path: Path = DEFAULT_RULES_PATH) -> list[WasteRule]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rules = [WasteRule(**row) for row in rows]
    ids = [rule.id for rule in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("규칙 ID가 중복되었습니다.")
    return rules


def load_schedule(path: Path = DEFAULT_SCHEDULE_PATH) -> dict:
    schedule = json.loads(path.read_text(encoding="utf-8"))
    neighborhoods = [
        name for zone in schedule["zones"] for name in zone["neighborhoods"]
    ]
    if len(neighborhoods) != len(set(neighborhoods)):
        raise ValueError("부산진구 동 이름이 여러 권역에 중복되었습니다.")
    return schedule


def neighborhood_names(schedule: dict | None = None) -> list[str]:
    data = schedule or load_schedule()
    return [name for zone in data["zones"] for name in zone["neighborhoods"]]


def schedule_guidance(
    rule: WasteRule,
    neighborhood: str,
    schedule: dict | None = None,
) -> ScheduleGuidance:
    data = schedule or load_schedule()
    zone = next(
        (zone for zone in data["zones"] if neighborhood in zone["neighborhoods"]),
        None,
    )
    if zone is None:
        raise ValueError(f"부산진구에서 지원하지 않는 동입니다: {neighborhood}")

    group = rule.schedule_group
    if group == "drop_off":
        days: list[str] = []
        method = "전용 수거함 이용"
        known = True
        note = "요일별 문전수거 대상이 아니라 전용 수거함의 운영 여부를 확인하세요."
    elif group == "verify" or group not in zone["days"]:
        days = []
        method = "부산진구 자원순환과 확인 필요"
        known = False
        note = "공식 요일표에서 이 세부 품목의 배출일을 확정할 수 없습니다."
    else:
        days = list(zone["days"][group])
        method = data["methods"][group]
        known = True
        note = "공동주택은 관리사무소의 별도 배출 기준을 우선 확인하세요."

    return ScheduleGuidance(
        neighborhood=neighborhood,
        zone=zone["label"],
        days=days,
        time=data["time"],
        place=data["place"],
        method=method,
        known=known,
        note=note,
        contact=data["contact"],
        source_title=data["source_title"],
        source_url=data["source_url"],
        effective_from=data["effective_from"],
        published_at=data["published_at"],
        verified_at=data["verified_at"],
    )


def _normalized(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        norm = np.linalg.norm(values)
        return values if norm == 0 else values / norm
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def corpus_fingerprint(rules: Sequence[WasteRule]) -> str:
    payload = json.dumps(
        [asdict(rule) for rule in rules], ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def safe_model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


class WasteSearchEngine:
    """한 임베딩 좌표계로 규칙을 색인하고 의미 검색을 수행한다."""

    def __init__(
        self,
        embedder: Embedder,
        model_name: str,
        rules: Sequence[WasteRule] | None = None,
        index_dir: Path = DEFAULT_INDEX_DIR,
        use_cache: bool = True,
    ) -> None:
        self.embedder = embedder
        self.model_name = model_name
        self.rules = list(load_rules() if rules is None else rules)
        if not self.rules:
            raise ValueError("검색할 분리배출 규칙이 없습니다.")
        self.index_dir = Path(index_dir)
        self.fingerprint = corpus_fingerprint(self.rules)
        self.index_path = self.index_dir / f"{safe_model_slug(model_name)}.npz"
        self.loaded_from_cache = False
        self.index = self._load_or_build_index(use_cache)

    def _load_or_build_index(self, use_cache: bool) -> np.ndarray:
        if use_cache and self.index_path.exists():
            try:
                with np.load(self.index_path, allow_pickle=False) as saved:
                    model = str(saved["model_name"].item())
                    fingerprint = str(saved["fingerprint"].item())
                    vectors = saved["vectors"]
                if (
                    model == self.model_name
                    and fingerprint == self.fingerprint
                    and len(vectors) == len(self.rules)
                ):
                    self.loaded_from_cache = True
                    return _normalized(vectors)
            except (KeyError, OSError, ValueError):
                pass

        vectors = self.embedder.encode([rule.search_text for rule in self.rules])
        index = _normalized(np.asarray(vectors))
        if index.ndim != 2 or len(index) != len(self.rules):
            raise ValueError("임베딩 모델이 문서 수와 맞지 않는 벡터를 반환했습니다.")
        if use_cache:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.index_path,
                vectors=index,
                model_name=np.array(self.model_name),
                fingerprint=np.array(self.fingerprint),
            )
        return index

    def semantic_search(self, query: str, k: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("검색어를 입력하세요.")
        q_vector = _normalized(np.asarray(self.embedder.encode(query)))
        if q_vector.ndim != 1 or q_vector.shape[0] != self.index.shape[1]:
            raise ValueError("질의와 인덱스의 임베딩 좌표계가 맞지 않습니다. 재색인하세요.")
        scores = self.index @ q_vector
        order = np.argsort(-scores)[: max(1, min(k, len(self.rules)))]
        return [
            SearchResult(self.rules[i], semantic_score=float(scores[i])) for i in order
        ]

    def keyword_search(self, query: str, k: int = 5) -> list[SearchResult]:
        query = query.strip().lower()
        if not query:
            raise ValueError("검색어를 입력하세요.")
        tokens = [token for token in re.findall(r"[0-9A-Za-z가-힣]+", query) if len(token) > 1]
        scores = []
        for i, rule in enumerate(self.rules):
            text = rule.search_text.lower()
            exact_alias = any(alias.lower() in query for alias in [rule.item, *rule.aliases])
            score = sum(text.count(token) for token in tokens) + (3 if exact_alias else 0)
            scores.append((score, i))
        order = sorted(scores, key=lambda row: (-row[0], row[1]))[: max(1, min(k, len(self.rules)))]
        return [SearchResult(self.rules[i], semantic_score=float(score)) for score, i in order]

    def search(
        self,
        query: str,
        k: int = 5,
        reranker: PairReranker | None = None,
        confidence_threshold: float = 0.35,
    ) -> SearchResponse:
        results = self.semantic_search(query, k=k)
        confident = results[0].semantic_score >= confidence_threshold
        if reranker is not None:
            pairs = [[query, result.rule.search_text] for result in results]
            rerank_scores = np.asarray(reranker.predict(pairs), dtype=float)
            if len(rerank_scores) != len(results):
                raise ValueError("재정렬 모델의 점수 수가 후보 수와 다릅니다.")
            results = [
                SearchResult(result.rule, result.semantic_score, float(score))
                for result, score in zip(results, rerank_scores)
            ]
            results.sort(
                key=lambda result: (
                    result.rerank_score
                    if result.rerank_score is not None
                    else float("-inf")
                ),
                reverse=True,
            )
        return SearchResponse(query, results, confident, confidence_threshold)


def make_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def make_reranker(model_name: str = DEFAULT_RERANKER_MODEL, device: str = "cpu"):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


def fallback_answer(
    response: SearchResponse,
    neighborhood: str | None = None,
    schedule: dict | None = None,
) -> str:
    """LLM 없이도 출처가 분명한 구조화 안내를 만든다."""
    if not response.confident or not response.results:
        return (
            "관련 규칙을 충분히 확신할 수 없습니다. 품목의 재질, 오염 여부와 "
            "현재 지역을 더 구체적으로 입력하고 실제 지자체 안내를 확인하세요."
        )
    rule = response.results[0].rule
    lines = [
        f"분류: {rule.category}\n"
        f"배출 방법: {rule.instructions}\n"
        f"주의사항: {rule.cautions}"
    ]
    if neighborhood:
        guidance = schedule_guidance(rule, neighborhood, schedule)
        if guidance.known and guidance.days:
            lines.append(f"배출 요일: {', '.join(guidance.days)} 저녁")
            lines.append(f"배출 시간: {guidance.time}")
            lines.append(f"배출 장소: {guidance.place}")
            lines.append(f"배출 용기: {guidance.method}")
        elif guidance.known:
            lines.append(f"수거 방식: {guidance.method}")
        else:
            lines.append(f"배출 일정: {guidance.method}")
        lines.append(f"지역: 부산진구 {neighborhood} ({guidance.zone})")
        lines.append(f"일정 참고: {guidance.note}")
    lines.append(
        f"품목 근거: {rule.source_title} · 게시/시행일 {rule.published_at} · "
        f"확인일 {rule.verified_at} · {rule.id}"
    )
    return "\n".join(lines)


def rag_answer(
    response: SearchResponse,
    neighborhood: str | None = None,
    schedule: dict | None = None,
    model: str = "waste-rag",
    base_url: str = "http://localhost:11434/v1",
) -> str:
    """검색 결과만 근거로 OpenAI 호환 로컬 모델에서 답변을 생성한다."""
    if not response.confident or not response.results:
        return fallback_answer(response, neighborhood, schedule)

    from openai import OpenAI

    context = "\n\n".join(
        f"[{result.rule.id}] {result.rule.search_text} 적용 지역: "
        f"{result.rule.jurisdiction}. 출처: {result.rule.source_title}. "
        f"게시/시행일: {result.rule.published_at}."
        for result in response.results[:3]
    )
    schedule_context = ""
    if neighborhood:
        guidance = schedule_guidance(response.results[0].rule, neighborhood, schedule)
        schedule_context = (
            f"\n\n[부산진구 일정]\n동: {neighborhood} ({guidance.zone})\n"
            f"요일: {', '.join(guidance.days) if guidance.days else '확정 불가 또는 거점수거'}\n"
            f"시간: {guidance.time}\n장소: {guidance.place}\n방법: {guidance.method}\n"
            f"주의: {guidance.note}\n출처: {guidance.source_title}"
        )
    client = OpenAI(base_url=base_url, api_key="not-needed")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "제공된 부산광역시·부산진구 공식 자료만 근거로 답한다. 품목 분류, "
                    "배출 방법, 주의사항, 근거 ID를 간결하게 포함한다. 규칙이 서로 "
                    "충돌하거나 일정 근거가 없으면 추측하지 말고 부산진구 자원순환과 "
                    "확인을 권한다. 공동주택은 관리사무소 기준을 우선한다고 밝힌다."
                ),
            },
            {
                "role": "user",
                "content": f"[규칙]\n{context}{schedule_context}\n\n[질문]\n{response.query}",
            },
        ],
        temperature=0.1,
        max_tokens=350,
    )
    return completion.choices[0].message.content or fallback_answer(
        response, neighborhood, schedule
    )
