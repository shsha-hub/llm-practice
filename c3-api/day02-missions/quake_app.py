"""
지진 알림 트리아지봇 — Streamlit 버전
tool_app.py와 달라지는 곳은 그릇뿐: 배열은 st.session_state, 로그는 st.caption
왕복 루프 자체는 quake_triage.py와 동일
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone

import streamlit as st
from openai import OpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel

client = OpenAI()
API_MODEL = "gpt-5.4-nano"


# ── 도구: 실시간 지진 데이터 조회 ──────────────────────────────────────
def get_recent_earthquakes(min_magnitude: float = 5.0, hours: int = 24) -> str:
    """최근 hours 시간 안에 규모 min_magnitude 이상 지진 목록을 USGS에서 가져온다 (키 불필요)"""
    start_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "format": "geojson",
        "starttime": start_time,
        "minmagnitude": min_magnitude,
        "orderby": "magnitude",
    }
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=10) as res:
            data = json.loads(res.read())
    except Exception as e:
        return f"지진 데이터 조회 실패: {e}"

    features = data.get("features", [])
    if not features:
        return f"최근 {hours}시간 동안 규모 {min_magnitude} 이상 지진 없음"

    lines = []
    for f in features[:10]:
        props = f["properties"]
        mag = props.get("mag")
        place = props.get("place")
        occurred = datetime.fromtimestamp(props.get("time", 0) / 1000, tz=timezone.utc)
        lines.append(f"규모 {mag} · {place} · {occurred.strftime('%Y-%m-%d %H:%M UTC')}")

    return f"총 {len(features)}건 감지:\n" + "\n".join(lines)


tools = [{
    "type": "function",
    "function": {
        "name": "get_recent_earthquakes",
        "description": "최근 일정 시간 안에 발생한 특정 규모 이상의 지진 목록을 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_magnitude": {"type": "number", "description": "조회할 최소 규모 (예: 5.0)"},
                "hours": {"type": "integer", "description": "몇 시간 전부터 조회할지"},
            },
            "required": ["min_magnitude", "hours"],
        },
    },
}]

TOOL_FUNCS = {"get_recent_earthquakes": get_recent_earthquakes}


class EarthquakeTriage(BaseModel):
    risk_level: str          # "낮음" / "중간" / "높음"
    summary: str
    biggest_quake: str
    quake_list: list[str]    # 도구가 반환한 순서(규모 큰 순) 그대로, 지진 하나당 한 줄


RISK_COLOR = {"낮음": "🟢", "중간": "🟡", "높음": "🔴"}


# ── Streamlit 화면 ──────────────────────────────────────────────────
st.title("최근 N시간 지진 조회봇")
st.caption(
    "- 지금 시점부터 원하는 만큼 거슬러 올라가 조회 (예: 최근 3시간 / 최근 3일 / 최근 1주일)\n"
    "- 규모 조건도 자유 지정 (예: 규모 3 이상 / 규모 6 이상)\n"
    "- '2011년 동일본대지진'처럼 특정 과거 날짜·역대 최대 지진 조회는 지원하지 않아요."
)

SYSTEM_PROMPT = (
    """
    너는 지진 조회봇이다. get_recent_earthquakes 도구는 '지금 시점부터 hours 시간만큼 거슬러 올라간 범위'와 '최소 규모(min_magnitude)'만 조건으로 받는다. 
    hours는 자유롭게 크게(예: 168=1주일, 720=1개월) 줘도 되지만, 반드시 지금을 기준으로 한 상대 시간이어야 한다. 
    '2011년 동일본대지진', '역대 최대 지진'처럼 특정 과거 날짜나 절대 시점을 묻는 질문에는 도구를 억지로 돌리지 말고, 
    이 도구는 지금부터의 상대 기간만 조회 가능하다고 정확히 안내해라.
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# dict(user/tool/최종 assistant)와 ChatCompletionMessage 객체(도구 호출 중간 assistant)가 배열에 섞여 있으므로, 
# role·content를 안전하게 꺼내는 헬퍼로 화면을 그린다
def _role_content(m):
    if isinstance(m, dict):
        return m.get("role"), m.get("content")
    return getattr(m, "role", None), getattr(m, "content", None)


for m in st.session_state.messages:
    role, content = _role_content(m)
    if role in ("user", "assistant") and content:
        with st.chat_message(role):
            st.markdown(content)

user_input = st.chat_input("예: 최근 3일간 규모 4 이상 지진 있었어?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # 1단계 — 도구 호출 왕복
        for _ in range(5):
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=st.session_state.messages,
                tools=tools,
                max_completion_tokens=500,
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                break
            st.session_state.messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = TOOL_FUNCS[tc.function.name](**args)
                st.caption(f"[도구] {tc.function.name}({args}) → {result[:60]}...")
                st.session_state.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        # 2단계 — 구조화 출력으로 위험도 판정
        triage_messages = st.session_state.messages + [{
            "role": "user",
            "content": "위 조회 결과를 바탕으로 위험도를 낮음/중간/높음 중 하나로 판정하고 요약해 줘.",
        }]
        try:
            rp = client.chat.completions.parse(
                model=API_MODEL,
                messages=triage_messages,
                response_format=EarthquakeTriage,
                max_completion_tokens=900,   # quake_list까지 다 채울 여유를 둠
            )
            result = rp.choices[0].message.parsed
        except LengthFinishReasonError:
            st.error("지진이 너무 많이 잡혀 답변이 중간에 잘렸어요. 조회 범위를 좁혀서 다시 물어봐 주세요 (예: 규모를 더 높이거나 기간을 줄이기).")
            st.stop()

        quake_lines = "\n".join(f"- {q}" for q in result.quake_list) or "(해당 없음)"
        answer = (
            f"{RISK_COLOR.get(result.risk_level, '⚪')} **위험도: {result.risk_level}**\n\n"
            f"{result.summary}\n\n"
            f"**가장 큰 지진** — {result.biggest_quake}\n\n"
            f"**전체 목록**\n{quake_lines}"
        )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})