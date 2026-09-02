"""Day 05 QLoRA 생활 도우미 Streamlit 데모."""

from __future__ import annotations

import re
import threading
import time

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from life_assistant.config import MODEL_ID, TASKS, generation_prompt, user_text
from life_assistant.inference import generate


TASK_LABELS = {
    "reply": "💬 곤란한 상황의 답장 작성기",
    "fridge": "🥗 냉장고 재료 소진 도우미",
}


st.set_page_config(page_title="QLoRA 생활 도우미", page_icon="🧩", layout="wide")
st.title("🧩 QLoRA 멀티 생활 도우미")
st.caption("Mi:dm 2B 4-bit base 하나에 두 개의 LoRA 어댑터를 바꿔 끼우는 실험 앱")


@st.cache_resource(show_spinner="4-bit base 모델과 두 LoRA 어댑터를 GPU에 올리는 중입니다…")
def load_model_bundle():
    """Base는 한 번만 로드하고 reply/fridge 어댑터를 이름으로 등록한다."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다. WSL의 GPU 연결을 확인하세요.")

    missing = [str(TASKS[name]["adapter"]) for name in TASKS if not TASKS[name]["adapter"].is_dir()]
    if missing:
        raise FileNotFoundError("학습된 어댑터가 없습니다: " + ", ".join(missing))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base,
        str(TASKS["reply"]["adapter"]),
        adapter_name="reply",
        local_files_only=True,
    )
    model.load_adapter(
        str(TASKS["fridge"]["adapter"]),
        adapter_name="fridge",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer, threading.Lock()


def format_checks(task: str, answer: str) -> dict[str, bool]:
    """사용자 입력의 정답을 모르는 상황에서 확인 가능한 형식 규칙."""
    if task == "reply":
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", answer))
        return {
            "2~3문장": 2 <= sentence_count <= 3,
            "180자 이내": len(answer) <= 180,
            "메타 설명 없음": not any(word in answer for word in ("답변:", "설명:", "예시:")),
        }

    step_count = len(re.findall(r"(?m)^\d+\. ", answer))
    extra_match = re.search(r"(?m)^추가 재료:\s*(.+)$", answer)
    extras = []
    if extra_match and extra_match.group(1).strip() != "없음":
        extras = [item.strip() for item in extra_match.group(1).split(",") if item.strip()]
    return {
        "필수 4개 섹션": all(
            marker in answer for marker in ("추천:", "추가 재료:", "조리:", "소진 재료:")
        ),
        "조리 3~4단계": 3 <= step_count <= 4,
        "추가 재료 2개 이하": len(extras) <= 2,
    }


def timed_generate(model, tokenizer, prompt: str, task: str) -> tuple[str, float]:
    started = time.perf_counter()
    answer = generate(model, tokenizer, prompt, task)
    return answer, time.perf_counter() - started


def run_comparison(task: str, row: dict) -> dict[str, tuple[str, float]]:
    model, tokenizer, lock = load_model_bundle()
    plain_prompt = generation_prompt(task, row, with_instruction=False)
    instructed_prompt = generation_prompt(task, row, with_instruction=True)

    # 캐시된 모델은 여러 Streamlit 세션이 공유하므로 adapter 전환과 생성을 직렬화한다.
    with lock:
        model.set_adapter(task)
        with model.disable_adapter():
            base = timed_generate(model, tokenizer, plain_prompt, task)
            prompted = timed_generate(model, tokenizer, instructed_prompt, task)
        model.set_adapter(task)
        adapter = timed_generate(model, tokenizer, plain_prompt, task)
    return {"base": base, "prompted": prompted, "adapter": adapter}


with st.sidebar:
    st.header("실험 설정")
    task = st.radio(
        "사용할 생활 도우미",
        list(TASK_LABELS),
        format_func=TASK_LABELS.get,
    )
    st.divider()
    if torch.cuda.is_available():
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_gib = torch.cuda.mem_get_info()[0] / 1024**3
        st.success(f"{torch.cuda.get_device_name(0)}")
        st.caption(f"VRAM 여유 {free_gib:.2f} / 전체 {total_gib:.2f} GiB")
    else:
        st.error("CUDA GPU 연결 안 됨")
    st.caption("첫 실행만 모델 로딩에 시간이 걸리며 이후에는 GPU 캐시를 재사용합니다.")


if task == "reply":
    st.subheader("상황을 입력하세요")
    st.info("실제 이름, 전화번호, 주소 등 개인정보는 입력하지 마세요.")
    with st.form("reply_form"):
        relation = st.text_input("상대와의 관계", "직장 동료")
        situation = st.text_area(
            "곤란한 상황",
            "다음 주 월요일까지 부탁받은 자료를 끝내기 어렵다",
            height=90,
        )
        intent = st.text_input("전하고 싶은 의도", "목요일까지 연기 요청")
        submitted = st.form_submit_button("세 모델로 답장 비교", type="primary", width="stretch")
    row = {"relation": relation.strip(), "situation": situation.strip(), "intent": intent.strip()}
    invalid = not all(row.values())
else:
    st.subheader("냉장고 속 재료를 입력하세요")
    st.warning("재료의 변질·알레르기 여부를 확인하고 육류·달걀은 충분히 익혀 드세요.")
    with st.form("fridge_form"):
        ingredients = st.text_area(
            "사용할 재료",
            "느타리버섯 한 팩, 달걀 2개, 양파 반 개",
            height=90,
        )
        condition = st.text_input("조리 조건", "1인분, 15분 이내")
        submitted = st.form_submit_button("세 모델로 레시피 비교", type="primary", width="stretch")
    row = {"ingredients": ingredients.strip(), "condition": condition.strip()}
    invalid = not all(row.values())


if submitted:
    if invalid:
        st.error("모든 입력란을 채워주세요.")
    else:
        try:
            with st.spinner("Base → Prompted base → QLoRA adapter 순서로 생성 중입니다…"):
                results = run_comparison(task, row)
            st.session_state["day05_results"] = {
                "task": task,
                "input": user_text(task, row),
                "results": results,
            }
        except Exception as exc:
            st.error(f"모델을 실행하지 못했습니다: {exc}")
            st.info("다른 Jupyter/Python 커널을 종료해 여유 VRAM을 3.2GiB 이상 확보해 주세요.")


saved = st.session_state.get("day05_results")
if saved and saved["task"] == task:
    st.divider()
    st.subheader("동일 입력 비교 결과")
    st.code(saved["input"], language=None)
    tabs = st.tabs(["Base", "Base + 상세 프롬프트", "QLoRA Adapter"])
    keys = ("base", "prompted", "adapter")
    for tab, key in zip(tabs, keys):
        answer, elapsed = saved["results"][key]
        with tab:
            st.write(answer)
            st.caption(f"생성 시간 {elapsed:.1f}초 · {len(answer)}자")
            checks = format_checks(task, answer)
            columns = st.columns(len(checks))
            for column, (label, passed) in zip(columns, checks.items()):
                column.metric(label, "통과" if passed else "미통과")
    st.caption(
        "형식 검사는 출력 구조만 확인합니다. 답장의 사회적 적절성이나 레시피의 안전·맛을 보장하지 않습니다."
    )
else:
    st.info("입력 후 비교 버튼을 누르면 동일한 base 모델에서 어댑터를 켜고 끈 결과를 확인할 수 있습니다.")
