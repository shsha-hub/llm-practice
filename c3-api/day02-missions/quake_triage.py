"""
실시간 지진 알림 트리아지봇
- 오늘 배운 두 파트를 이어붙인 구조: ① 도구 호출로 실시간 데이터 조회 → ② 구조화 출력으로 위험도 판정
- API: USGS Earthquake Catalog (키 불필요, 무료) — https://earthquake.usgs.gov/fdsnws/event/1/
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()
API_MODEL = "gpt-5.4-nano"


# ── ① 도구: 실시간 지진 데이터 조회 (get_weather와 완전히 같은 패턴) ──────────
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
    for f in features[:10]:  # 상위 10개까지만
        props = f["properties"]
        mag = props.get("mag")
        place = props.get("place")
        # USGS 시간은 밀리초 단위 UTC epoch
        occurred = datetime.fromtimestamp(props.get("time", 0) / 1000, tz=timezone.utc)
        lines.append(f"규모 {mag} · {place} · {occurred.strftime('%Y-%m-%d %H:%M UTC')}")

    return f"총 {len(features)}건 감지:\n" + "\n".join(lines)


# ── 함수 설명서 — 오늘 배운 JSON 스키마 문법 그대로 ───────────────────────
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


# ── ② 구조화 출력 — 오전에 배운 3단(Pydantic) 그대로 ──────────────────────
class EarthquakeTriage(BaseModel):
    risk_level: str         # "낮음" / "중간" / "높음"
    summary: str            # 한두 문장 요약
    biggest_quake: str      # 가장 큰 지진 한 줄 설명


def triage(user_input: str) -> EarthquakeTriage:
    messages = [{"role": "user", "content": user_input}]

    # 1단계 — 도구 호출 왕복 (오늘 배운 네 걸음, tool_chat.py와 동일)
    for _ in range(5):
        response = client.chat.completions.create(
            model=API_MODEL, messages=messages, tools=tools, max_completion_tokens=500
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOL_FUNCS[tc.function.name](**args)
            print(f"  [도구] {tc.function.name}({args}) → {result[:60]}...")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # 2단계 — 조회 결과를 바탕으로 구조화된 위험도 판정 요청 (오전 3단 패턴)
    messages.append({
        "role": "user",
        "content": "위 조회 결과를 바탕으로 위험도를 낮음/중간/높음 중 하나로 판정하고 요약해 줘.",
    })
    rp = client.chat.completions.parse(
        model=API_MODEL,
        messages=messages,
        response_format=EarthquakeTriage,
        max_completion_tokens=400,
    )
    return rp.choices[0].message.parsed


if __name__ == "__main__":
    result = triage("최근 24시간 동안 규모 5 이상 지진 있었어? 위험한지 판단해 줘.")
    print("\n=== 최종 판정 ===")
    print(f"위험도 : {result.risk_level}")
    print(f"요약   : {result.summary}")
    print(f"최대 지진 : {result.biggest_quake}")