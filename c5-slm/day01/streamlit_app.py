"""Mission 02: seed별 로컬 LLM 성공률 평가 대시보드."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import torch

from llm_evaluator import (
    DEFAULT_MODEL_ID,
    load_model,
    run_seed_experiment,
    success_rates,
)


DEFAULT_PROMPT = """한국어 2행시를 작성하세요.

예시:
제시어: 바다
바: 바람이 살며시 불어오고
다: 다정한 파도가 반겨준다

새 과제:
제시어: 점심
주제: 즐거운 점심시간
규칙: 점과 심으로 시작하는 두 줄만 출력하고 제목이나 설명은 쓰지 마세요.
답변:
""".strip()


st.set_page_config(page_title="로컬 LLM 성공률 평가기", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Qwen 모델을 GPU에 올리는 중입니다...")
def get_model():
    """Streamlit 재실행마다 모델을 다시 로드하지 않도록 캐시한다."""
    return load_model(DEFAULT_MODEL_ID)


def generation_options(
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
) -> dict:
    options = {
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
    }
    if do_sample:
        options.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    return options


st.title("🧪 로컬 LLM seed별 성공률 평가기")
st.caption("한 번의 성공 여부가 아니라 여러 seed에서의 형식·주제 준수율을 측정합니다.")

with st.sidebar:
    st.header("생성 설정")
    st.code(DEFAULT_MODEL_ID, language=None)
    do_sample = st.toggle("샘플링 사용", value=True)
    temperature = st.slider(
        "temperature",
        0.1,
        1.5,
        0.7,
        0.1,
        disabled=not do_sample,
    )
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05, disabled=not do_sample)
    top_k = st.slider("top_k", 0, 100, 20, 5, disabled=not do_sample)
    repetition_penalty = st.slider("repetition_penalty", 1.0, 1.4, 1.0, 0.05)
    max_new_tokens = st.slider("max_new_tokens", 20, 200, 80, 10)

    st.header("반복 설정")
    start_seed = st.number_input("시작 seed", min_value=0, value=0, step=1)
    seed_count = st.slider("실행 횟수", 1, 20, 5)

if torch.cuda.is_available():
    st.success(f"CUDA 사용 가능 · {torch.cuda.get_device_name(0)}")
else:
    st.error("CUDA를 사용할 수 없습니다. 강의 때 사용한 Python 가상환경으로 실행하세요.")

prompt = st.text_area("평가 프롬프트", value=DEFAULT_PROMPT, height=300)
run_button = st.button(
    "seed 실험 실행",
    type="primary",
    disabled=not torch.cuda.is_available(),
    use_container_width=True,
)

if run_button:
    if not prompt.strip():
        st.warning("프롬프트를 입력하세요.")
    else:
        tokenizer, model = get_model()
        seeds = range(int(start_seed), int(start_seed) + seed_count)
        options = generation_options(
            do_sample,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_new_tokens,
        )
        progress = st.progress(0, text="seed 실험 준비 중")

        def update_progress(completed: int, total: int) -> None:
            progress.progress(completed / total, text=f"생성 중: {completed}/{total}")

        rows = run_seed_experiment(
            model,
            tokenizer,
            prompt,
            seeds,
            generation_options=options,
            progress_callback=update_progress,
        )
        progress.empty()
        st.session_state["evaluation_results"] = pd.DataFrame(rows)
        st.session_state["last_options"] = options
        st.session_state["result_version"] = st.session_state.get("result_version", 0) + 1

if "evaluation_results" in st.session_state:
    st.subheader("seed별 결과와 사람 평가")
    st.info(
        "`manual_topic_pass`는 답변을 읽고 직접 체크하세요. "
        "`keyword_hit`는 단어 포함 여부일 뿐 의미 평가가 아닙니다."
    )

    edited = st.data_editor(
        st.session_state["evaluation_results"],
        key=f"result_editor_{st.session_state.get('result_version', 0)}",
        hide_index=True,
        use_container_width=True,
        disabled=[
            "seed",
            "answer",
            "new_tokens",
            "seconds",
            "tok_per_s",
            "core_pattern_pass",
            "strict_format_pass",
            "keyword_hit",
            "matched_keywords",
        ],
        column_config={
            "seed": st.column_config.NumberColumn("seed", format="%d"),
            "answer": st.column_config.TextColumn("생성 답변", width="large"),
            "core_pattern_pass": st.column_config.CheckboxColumn("점→심 패턴"),
            "strict_format_pass": st.column_config.CheckboxColumn("정확히 두 줄"),
            "keyword_hit": st.column_config.CheckboxColumn("주제 키워드"),
            "matched_keywords": "발견 키워드",
            "manual_topic_pass": st.column_config.CheckboxColumn(
                "사람 주제 평가",
                help="답변이 실제로 점심 주제를 유지하면 체크합니다.",
            ),
            "new_tokens": "생성 토큰",
            "seconds": "시간(초)",
            "tok_per_s": "tok/s",
        },
    )

    rows_for_rates = edited.to_dict(orient="records")
    rates = success_rates(rows_for_rates)
    metric_columns = st.columns(4)
    metric_columns[0].metric("핵심 패턴", f"{rates['core_pattern_rate']}%")
    metric_columns[1].metric("엄격한 두 줄", f"{rates['strict_format_rate']}%")
    metric_columns[2].metric("사람 주제 평가", f"{rates['manual_topic_rate']}%")
    metric_columns[3].metric("전체 성공", f"{rates['overall_rate']}%")

    st.caption("전체 성공 = 엄격한 두 줄 형식 통과 + 사람이 점심 주제 통과로 평가")
    st.download_button(
        "CSV로 결과 저장",
        data=edited.to_csv(index=False).encode("utf-8-sig"),
        file_name="seed_evaluation_results.csv",
        mime="text/csv",
    )

    with st.expander("마지막 생성 설정"):
        st.json(st.session_state.get("last_options", {}))
else:
    st.subheader("진행 방법")
    st.markdown(
        """
1. 왼쪽에서 생성 옵션과 실행 횟수를 정합니다.
2. 프롬프트를 확인하고 **seed 실험 실행**을 누릅니다.
3. 각 답변을 읽고 `사람 주제 평가`를 체크합니다.
4. 자동 형식 성공률과 사람이 판단한 전체 성공률을 비교합니다.
"""
    )
