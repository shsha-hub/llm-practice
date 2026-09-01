"""Day 04 미니 프로젝트: 버려도 될까?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# Streamlit은 실행한 스크립트의 폴더를 import 경로에 넣는다.
# 따라서 저장소 루트와 day04 폴더 어디에서 실행해도 core를 찾을 수 있다.
from core import (
    DEFAULT_RERANKER_MODEL,
    WasteSearchEngine,
    fallback_answer,
    load_rules,
    load_schedule,
    make_embedder,
    make_reranker,
    neighborhood_names,
    rag_answer,
    schedule_guidance,
)


EMBEDDING_MODELS = {
    "ko-sroberta (768차원·경량)": "jhgan/ko-sroberta-multitask",
    "bge-m3 (1024차원·고성능)": "BAAI/bge-m3",
}

st.set_page_config(page_title="버려도 될까?", page_icon="♻️", layout="wide")
st.title("♻️ 버려도 될까?")
st.caption("공식 자료와 임베딩 검색을 이용한 부산진구 생활폐기물 배출 도우미")
st.warning(
    "부산광역시·환경부 품목 지침과 부산진구가 2025-02-19 게시한 요일표를 "
    "바탕으로 합니다. 공동주택은 관리사무소 기준이 다를 수 있으며, 실제 배출 "
    "전에는 아래 공식 출처의 최신 내용을 다시 확인하세요."
)

schedule = load_schedule()


@st.cache_resource(show_spinner="임베딩 모델을 불러오고 규칙을 색인하는 중입니다…")
def get_engine(model_name: str) -> WasteSearchEngine:
    return WasteSearchEngine(make_embedder(model_name, device="cpu"), model_name)


@st.cache_resource(show_spinner="한국어 재정렬 모델을 불러오는 중입니다…")
def get_reranker():
    return make_reranker(DEFAULT_RERANKER_MODEL, device="cpu")


with st.sidebar:
    st.header("검색 설정")
    model_label = st.selectbox("임베딩 모델", list(EMBEDDING_MODELS))
    use_reranker = st.toggle(
        "Cross-encoder 재정렬",
        value=True,
        help="유리병/깨진 유리처럼 비슷한 규칙을 구분하므로 실제 안내에서는 켜는 것을 권장합니다.",
    )
    candidate_count = st.slider("1차 검색 후보 수", 3, 10, 5)
    threshold = st.slider(
        "최소 의미 유사도",
        0.0,
        1.0,
        0.35,
        0.05,
        help="최상위 검색 점수가 이 값보다 낮으면 답변을 보류합니다.",
    )
    use_llm = st.toggle("로컬 LLM으로 답변 생성", value=False)
    if use_llm:
        ollama_url = st.text_input("OpenAI 호환 API", "http://localhost:11434/v1")
        ollama_model = st.text_input("Ollama 모델", "waste-rag")
    else:
        ollama_url = "http://localhost:11434/v1"
        ollama_model = "waste-rag"

model_name = EMBEDDING_MODELS[model_label]
try:
    engine = get_engine(model_name)
except Exception as exc:
    st.error(f"임베딩 모델을 준비하지 못했습니다: {exc}")
    st.info("네트워크가 가능한 환경에서 모델을 한 번 내려받은 뒤 다시 실행하세요.")
    st.stop()

cache_message = "저장된 인덱스 재사용" if engine.loaded_from_cache else "새 인덱스 생성"
st.sidebar.caption(
    f"{cache_message} · {engine.index.shape[0]}개 규칙 × {engine.index.shape[1]}차원"
)
st.sidebar.caption("모델 또는 데이터가 바뀌면 새 좌표계로 자동 재색인합니다.")

assistant_tab, compare_tab, schedule_tab, rules_tab = st.tabs(
    ["배출 방법 묻기", "검색 방식 비교", "동별 배출 일정", "공식 근거 데이터"]
)

with assistant_tab:
    if not use_reranker:
        st.warning(
            "재정렬을 끈 상태는 검색 성능 비교용입니다. 비슷한 품목을 잘못 선택할 "
            "수 있으므로 실제 배출 안내에는 재정렬을 켜세요."
        )
    neighborhood = st.selectbox(
        "거주 동",
        neighborhood_names(schedule),
        help="단독주택·문전수거 기준입니다. 공동주택은 관리사무소에 확인하세요.",
    )
    query = st.text_area(
        "버릴 물건의 재질과 상태를 자연스럽게 적어 주세요",
        "배달 짬뽕을 먹고 빨간 기름이 씻기지 않는 플라스틱 통은 어떻게 버려요?",
        height=90,
    )
    search_button = st.button("배출 방법 찾기", type="primary", width="stretch")
    if search_button:
        try:
            reranker = get_reranker() if use_reranker else None
            response = engine.search(
                query,
                k=candidate_count,
                reranker=reranker,
                confidence_threshold=threshold,
            )
            st.session_state["waste_response"] = response
            st.session_state["waste_neighborhood"] = neighborhood
        except Exception as exc:
            st.error(f"검색하지 못했습니다: {exc}")

    if response := st.session_state.get("waste_response"):
        answer_neighborhood = st.session_state.get("waste_neighborhood", neighborhood)
        top_score = response.results[0].semantic_score if response.results else 0.0
        if response.confident:
            st.success(f"관련 규칙을 찾았습니다 · 1차 유사도 {top_score:.3f}")
        else:
            st.warning(f"최상위 유사도 {top_score:.3f}가 기준 {response.confidence_threshold:.2f}보다 낮아 답변을 보류합니다.")

        if use_llm and response.confident:
            try:
                with st.spinner("검색된 규칙만 근거로 답변을 생성하는 중입니다…"):
                    answer = rag_answer(
                        response,
                        answer_neighborhood,
                        schedule,
                        model=ollama_model,
                        base_url=ollama_url,
                    )
            except Exception as exc:
                st.warning(f"로컬 LLM에 연결하지 못해 규칙 원문으로 안내합니다: {exc}")
                answer = fallback_answer(response, answer_neighborhood, schedule)
        else:
            answer = fallback_answer(response, answer_neighborhood, schedule)
        st.code(answer, language=None)

        st.subheader("검색 근거")
        for rank, result in enumerate(response.results[:3], 1):
            score_text = f"의미 유사도 {result.semantic_score:.3f}"
            if result.rerank_score is not None:
                score_text += f" · 재정렬 점수 {result.rerank_score:+.3f}"
            with st.expander(f"{rank}. {result.rule.item} · {score_text}"):
                st.write(result.rule.instructions)
                st.caption(
                    f"{result.rule.id} · 게시/시행일 {result.rule.published_at} · "
                    f"확인일 {result.rule.verified_at}"
                )
                st.link_button("품목 기준 공식 출처", result.rule.source_url)

with compare_tab:
    st.subheader("키워드 검색과 의미 검색 비교")
    compare_query = st.text_input(
        "비교할 질문", "택배에 들어 있던 깨끗한 하얀 완충재를 버리고 싶어요"
    )
    if st.button("두 방식 비교"):
        try:
            keyword_results = engine.keyword_search(compare_query, k=5)
            semantic_results = engine.semantic_search(compare_query, k=5)
            left, right = st.columns(2)
            with left:
                st.markdown("**단어 겹침 검색**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "순위": range(1, 6),
                            "품목": [r.rule.item for r in keyword_results],
                            "점수": [r.semantic_score for r in keyword_results],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            with right:
                st.markdown("**임베딩 의미 검색**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "순위": range(1, 6),
                            "품목": [r.rule.item for r in semantic_results],
                            "점수": [round(r.semantic_score, 3) for r in semantic_results],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"비교하지 못했습니다: {exc}")

with schedule_tab:
    st.subheader("부산진구 동별 문전수거 일정")
    schedule_neighborhood = st.selectbox(
        "일정을 확인할 동", neighborhood_names(schedule), key="schedule_neighborhood"
    )
    rows = []
    seen_groups = set()
    for rule in load_rules():
        if rule.schedule_group in seen_groups or rule.schedule_group in {"verify", "drop_off"}:
            continue
        seen_groups.add(rule.schedule_group)
        guidance = schedule_guidance(rule, schedule_neighborhood, schedule)
        rows.append(
            {
                "분류 예시": rule.category,
                "배출 요일": ", ".join(guidance.days),
                "배출 방법": guidance.method,
                "시간": guidance.time,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"{schedule['housing_scope']} · {schedule['place']} · "
        f"금·토요일 배출 금지 · {schedule['contact']}"
    )
    st.link_button("부산진구 공식 요일표", schedule["source_url"])
    with st.expander("자료 시점과 적용 범위"):
        for note in schedule["notes"]:
            st.write(f"- {note}")

with rules_tab:
    st.subheader("부산광역시·환경부 공식 자료 기반 규칙")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": rule.id,
                    "품목": rule.item,
                    "분류": rule.category,
                    "다른 표현": ", ".join(rule.aliases),
                    "근거": rule.source_title,
                    "게시·시행일": rule.published_at,
                    "확인일": rule.verified_at,
                }
                for rule in load_rules()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "품목별 기준은 환경부 생활폐기물 분리배출 누리집과 부산광역시 자료, 요일·시간은 부산진구 공식 안내를 사용합니다."
        "오래된 부산시 자료만으로 일정을 추정하지 않으며, 불명확한 세부 품목은 문의하도록 안내합니다."
    )
