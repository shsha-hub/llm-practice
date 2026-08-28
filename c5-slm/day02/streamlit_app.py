"""Day 02 미니 과제: 내 PC에 맞는 로컬 LLM 선택기."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ollama_benchmark import (
    DEFAULT_OLLAMA_URL,
    ModelInfo,
    OllamaClient,
    OllamaError,
    benchmark_model,
    generate_with_fallback,
    recommend_models,
)


DEFAULT_PROMPT = "재택근무의 장점 세 가지를 한국어로 짧게 알려 줘."

st.set_page_config(page_title="로컬 LLM 선택기", page_icon="⚖️", layout="wide")
st.title("⚖️ 내 PC에 맞는 로컬 LLM 선택기")
st.caption("같은 질문으로 양자화 모델의 속도와 자원 사용량을 직접 비교합니다.")

with st.sidebar:
    st.header("Ollama 설정")
    ollama_url = st.text_input("서버 주소", value=DEFAULT_OLLAMA_URL)
    st.caption("터미널에서 `ollama serve`를 먼저 실행하세요.")
    refresh = st.button("모델 목록 새로고침", use_container_width=True)

    st.header("생성 설정")
    num_predict = st.slider("최대 생성 토큰", 20, 300, 120, 10)
    temperature = st.slider("temperature", 0.0, 1.5, 0.0, 0.1)


@st.cache_data(ttl=15, show_spinner=False)
def load_models(base_url: str) -> list[ModelInfo]:
    return OllamaClient(base_url).list_models()


if refresh:
    load_models.clear()

try:
    models = load_models(ollama_url)
    connection_error = "" if models else "설치된 Ollama 모델이 없습니다."
except OllamaError as exc:
    models = []
    connection_error = str(exc)

if connection_error:
    st.error(connection_error)
    st.info("Ollama 서버를 실행하고 모델을 받은 다음 왼쪽의 새로고침 버튼을 누르세요.")
    st.code("ollama serve\nollama list", language="bash")
    st.stop()

model_by_name = {model.name: model for model in models}
model_names = list(model_by_name)

st.success(f"Ollama 연결됨 · 설치 모델 {len(models)}개")
with st.expander("설치된 모델 정보"):
    st.dataframe(
        [
            {
                "모델": model.name,
                "양자화": model.quantization,
                "파일 크기(GB)": model.size_gb,
            }
            for model in models
        ],
        hide_index=True,
        use_container_width=True,
    )

compare_tab, fallback_tab = st.tabs(["모델 비교", "Fallback 실험"])

with compare_tab:
    default_selection = model_names[: min(3, len(model_names))]
    selected_names = st.multiselect(
        "비교할 모델",
        options=model_names,
        default=default_selection,
        help="같은 base 모델의 Q2/Q4/Q8 버전을 선택하면 차이가 잘 보입니다.",
    )
    prompt = st.text_area("비교 프롬프트", value=DEFAULT_PROMPT, height=110)
    run_compare = st.button(
        "벤치마크 실행",
        type="primary",
        use_container_width=True,
        disabled=not selected_names,
    )

    if run_compare:
        if not prompt.strip():
            st.warning("프롬프트를 입력하세요.")
        else:
            client = OllamaClient(ollama_url)
            results = []
            progress = st.progress(0, text="벤치마크 준비 중")
            for index, name in enumerate(selected_names, start=1):
                progress.progress(
                    (index - 1) / len(selected_names),
                    text=f"{name} 실행 중 ({index}/{len(selected_names)})",
                )
                results.append(
                    benchmark_model(
                        client,
                        model_by_name[name],
                        prompt,
                        num_predict=num_predict,
                        temperature=temperature,
                    )
                )
            progress.progress(1.0, text="완료")
            progress.empty()
            st.session_state["benchmark_results"] = results
            st.session_state["benchmark_prompt"] = prompt

    if results := st.session_state.get("benchmark_results"):
        st.subheader("비교 결과")
        table_rows = [
            {
                "모델": row["name"],
                "양자화": row["quantization"],
                "상태": row["status"],
                "파일(GB)": row["size_gb"],
                "VRAM(GB)": row["vram_gb"],
                "시간(초)": row["seconds"],
                "생성 토큰": row["tokens"],
                "tok/s": row["tok_per_s"],
                "오류": row["error"],
            }
            for row in results
        ]
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        recommendations = recommend_models(results)
        if recommendations:
            columns = st.columns(3)
            columns[0].metric("가장 빠름", recommendations["fastest"])
            columns[1].metric("가장 적은 자원", recommendations["smallest"])
            columns[2].metric("속도·자원 균형", recommendations["balanced"])
            st.caption("균형 추천은 답변 정확도가 아닌 생성 속도 60%와 자원 효율 40% 기준입니다.")
        else:
            st.warning("성공한 모델이 없습니다. 오류 메시지와 Ollama 상태를 확인하세요.")

        st.subheader("모델별 답변")
        for row in results:
            with st.expander(f"{row['name']} · {row['status']}"):
                st.write(row["answer"] or row["error"])

        export_data = {
            "prompt": st.session_state.get("benchmark_prompt", ""),
            "settings": {
                "num_predict": num_predict,
                "temperature": temperature,
            },
            "recommendations": recommendations,
            "results": results,
        }
        st.download_button(
            "JSON으로 결과 저장",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name="ollama_benchmark_results.json",
            mime="application/json",
        )

with fallback_tab:
    st.write("큰 모델부터 시도하고 실패하면 지정한 순서대로 다음 모델을 실행합니다.")
    primary_name = st.selectbox("우선 모델", options=model_names)
    fallback_names = st.multiselect(
        "Fallback 모델 (선택 순서대로 시도)",
        options=[name for name in model_names if name != primary_name],
    )
    fallback_prompt = st.text_area(
        "Fallback 실험 프롬프트",
        value="파이썬의 리스트와 튜플 차이를 두 문장으로 설명해 줘.",
        height=100,
    )
    run_fallback = st.button(
        "Fallback 실행",
        type="primary",
        use_container_width=True,
    )

    if run_fallback:
        if not fallback_prompt.strip():
            st.warning("프롬프트를 입력하세요.")
        else:
            ordered_models = [model_by_name[primary_name]] + [
                model_by_name[name] for name in fallback_names
            ]
            with st.spinner("모델을 순서대로 시도하는 중..."):
                result, attempts = generate_with_fallback(
                    OllamaClient(ollama_url),
                    ordered_models,
                    fallback_prompt,
                    num_predict=num_predict,
                    temperature=temperature,
                )
            st.session_state["fallback_result"] = result
            st.session_state["fallback_attempts"] = attempts

    if "fallback_attempts" in st.session_state:
        result = st.session_state["fallback_result"]
        attempts = st.session_state["fallback_attempts"]
        if result:
            st.success(f"`{result['name']}` 모델로 생성에 성공했습니다.")
            st.write(result["answer"])
        else:
            st.error("모든 모델 실행에 실패했습니다.")
        st.dataframe(
            [
                {
                    "시도 순서": index,
                    "모델": row["name"],
                    "상태": row["status"],
                    "시간(초)": row["seconds"],
                    "오류": row["error"],
                }
                for index, row in enumerate(attempts, start=1)
            ],
            hide_index=True,
            use_container_width=True,
        )
