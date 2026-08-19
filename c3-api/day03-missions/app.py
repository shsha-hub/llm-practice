"""날씨 기반 맞춤형 하루 계획 도우미 Streamlit 앱."""

from __future__ import annotations

import hashlib
import os
import uuid

import streamlit as st
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from weather_planner import API_MODEL, DailyPlan, WeatherPlanner


st.set_page_config(
    page_title="날씨 기반 하루 계획 도우미",
    page_icon="🌤️",
    layout="centered",
)


def get_api_key() -> str | None:
    """환경변수를 우선하고, 없으면 Streamlit secrets에서 API 키를 읽는다."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def render_plan(plan_data: dict) -> None:
    """저장된 딕셔너리를 검증하고 하루 계획 카드로 표시한다."""
    plan = DailyPlan.model_validate(plan_data)
    st.markdown(f"### 📍 {plan.city}")
    st.info(f"**현재 날씨**\n\n{plan.weather_summary}")

    for activity in plan.activities:
        with st.container(border=True):
            st.markdown(f"#### {activity.time} · {activity.title}")
            st.markdown(f"**장소**: {activity.location}")
            st.markdown(f"**추천 이유**: {activity.reason}")
            preparations = ", ".join(activity.preparation) or "별도 준비물 없음"
            st.markdown(f"**준비물**: {preparations}")

    st.warning(f"**주의사항**\n\n{plan.caution}")


def plan_to_speech_text(plan: DailyPlan) -> str:
    """추가 모델 호출 없이 구조화 계획을 TTS용 문장으로 바꾼다."""
    activities = " ".join(
        f"{item.time}에는 {item.title}을 추천합니다. 장소는 {item.location}입니다."
        for item in plan.activities
    )
    return (
        f"{plan.city} 맞춤 계획입니다. 현재 날씨는 {plan.weather_summary}입니다. "
        f"{activities} 주의사항은 {plan.caution}입니다."
    )


def friendly_error(exc: Exception) -> str:
    """외부 서비스 오류를 사용자가 이해하기 쉬운 메시지로 변환한다."""
    if isinstance(exc, AuthenticationError):
        return "OpenAI API 키를 확인해 주세요."
    if isinstance(exc, RateLimitError):
        return "API 사용 한도에 도달했습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return "OpenAI API 연결이 원활하지 않습니다. 잠시 후 다시 시도해 주세요."
    return str(exc) or "요청 처리 중 알 수 없는 오류가 발생했습니다."


def render_usage_metrics(
    planner: WeatherPlanner,
    input_slot,
    output_slot,
    cached_slot,
    total_slot,
) -> None:
    """현재 누적 Responses 사용량으로 사이드바 지표를 갱신한다."""
    usage = planner.usage()
    input_slot.metric("입력", f"{usage['input_tokens']:,}")
    output_slot.metric("출력", f"{usage['output_tokens']:,}")
    cached_slot.metric("캐시", f"{usage['cached_tokens']:,}")
    total_slot.metric("합계", f"{usage['total_tokens']:,}")


api_key = get_api_key()

st.title("🌤️ 날씨 기반 하루 계획 도우미")
st.caption("현재 날씨와 원하는 시간·활동을 바탕으로 실천 가능한 계획을 만들어 드립니다.")

if not api_key:
    st.error("`OPENAI_API_KEY`가 설정되어 있지 않습니다.")
    st.code('export OPENAI_API_KEY="YOUR_API_KEY"', language="bash")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

api_key_fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:12]
safety_identifier = hashlib.sha256(
    f"weather-planner:{st.session_state.session_id}".encode()
).hexdigest()

if (
    "planner" not in st.session_state
    or st.session_state.get("api_key_fingerprint") != api_key_fingerprint
):
    st.session_state.planner = WeatherPlanner(
        api_key=api_key,
        safety_identifier=safety_identifier,
    )
    st.session_state.api_key_fingerprint = api_key_fingerprint
    st.session_state.messages = []
    st.session_state.last_speech_text = None
    st.session_state.audio_bytes = None

planner: WeatherPlanner = st.session_state.planner

with st.sidebar:
    st.header("설정")
    st.caption(f"모델: `{API_MODEL}`")
    stream_enabled = st.toggle(
        "AI 자연어 브리핑 추가",
        value=True,
        help="계획 카드 아래에 Responses 스트리밍 설명을 추가합니다.",
    )

    if st.button("새 대화 시작", use_container_width=True):
        planner.reset()
        st.session_state.messages = []
        st.session_state.last_speech_text = None
        st.session_state.audio_bytes = None
        st.rerun()

    st.divider()
    st.subheader("Responses 사용량")
    col1, col2 = st.columns(2)
    input_metric = col1.empty()
    output_metric = col2.empty()
    cached_metric = col1.empty()
    total_metric = col2.empty()
    render_usage_metrics(
        planner,
        input_metric,
        output_metric,
        cached_metric,
        total_metric,
    )
    st.caption("Moderation과 TTS 사용량은 위 통계에 포함되지 않습니다.")

with st.expander("이렇게 요청해 보세요", expanded=not st.session_state.messages):
    st.markdown(
        """
- `부산에서 오늘 저녁 9시부터 2시간 동안 가볍게 운동하고 싶어.`
- `비가 오면 실내 활동으로만 구성해 줘.`
- `같은 시간 안에서 비용이 거의 들지 않는 계획으로 바꿔 줘.`
        """
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif message.get("error"):
            st.error(message["error"])
        else:
            render_plan(message["plan"])
            if message.get("briefing"):
                st.markdown("##### AI 브리핑")
                st.markdown(message["briefing"])

prompt = st.chat_input("지역, 가능한 시간, 원하는 활동을 알려 주세요.")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.audio_bytes = None

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("현재 날씨를 확인하고 계획을 만드는 중입니다..."):
                plan = planner.ask(prompt)

            plan_data = plan.model_dump()
            render_plan(plan_data)
            briefing = None

            if stream_enabled:
                st.markdown("##### AI 브리핑")
                placeholder = st.empty()
                visible_chunks: list[str] = []

                def show_delta(delta: str) -> None:
                    visible_chunks.append(delta)
                    placeholder.markdown("".join(visible_chunks) + "▌")

                briefing = planner.stream_briefing(on_delta=show_delta)
                placeholder.markdown(briefing)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "plan": plan_data,
                    "briefing": briefing,
                }
            )
            st.session_state.last_speech_text = briefing or plan_to_speech_text(plan)
            render_usage_metrics(
                planner,
                input_metric,
                output_metric,
                cached_metric,
                total_metric,
            )
        except Exception as exc:
            error_message = friendly_error(exc)
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "error": error_message}
            )
            render_usage_metrics(
                planner,
                input_metric,
                output_metric,
                cached_metric,
                total_metric,
            )

if st.session_state.get("last_speech_text"):
    st.divider()
    st.subheader("🔊 음성 브리핑")
    st.caption("이 음성은 AI로 생성됩니다.")

    if st.button("최신 계획을 음성으로 만들기"):
        try:
            with st.spinner("음성 파일을 생성하는 중입니다..."):
                st.session_state.audio_bytes = planner.create_speech(
                    st.session_state.last_speech_text
                )
        except Exception as exc:
            st.error(friendly_error(exc))

    if st.session_state.get("audio_bytes"):
        st.audio(st.session_state.audio_bytes, format="audio/mp3")
        st.download_button(
            "MP3 다운로드",
            data=st.session_state.audio_bytes,
            file_name="weather_plan.mp3",
            mime="audio/mpeg",
        )
