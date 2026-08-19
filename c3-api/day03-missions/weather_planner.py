"""날씨 기반 하루 계획 도우미의 API 및 도메인 로직."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from openai import OpenAI
from pydantic import BaseModel, Field


API_MODEL = "gpt-5.6-luna"
MODERATION_MODEL = "omni-moderation-latest"
SPEECH_MODEL = "tts-1"


class Activity(BaseModel):
    time: str = Field(description="활동 시작 시각 또는 시간대")
    title: str = Field(description="활동 이름")
    location: str = Field(description="추천 장소 또는 실내/실외 구분")
    reason: str = Field(description="날씨와 사용자 조건을 고려한 추천 이유")
    preparation: list[str] = Field(description="필요한 준비물")


class DailyPlan(BaseModel):
    city: str
    weather_summary: str
    activities: list[Activity]
    caution: str = Field(description="날씨 또는 안전 관련 주의사항")


PLANNER_INSTRUCTIONS = """
당신은 날씨 기반 하루 계획 도우미입니다.
사용자의 지역, 가능한 시간, 선호 활동을 반영해 현실적인 계획을 만드세요.
반드시 도구가 반환한 현재 날씨만 날씨 판단의 근거로 사용하세요.
강수, 폭염, 한파, 강풍 가능성이 있으면 무리한 야외 활동을 피하고 대안을 제안하세요.
사용자가 밝히지 않은 취향이나 건강 상태를 사실처럼 단정하지 마세요.
응답은 요청된 DailyPlan 스키마를 정확히 따르세요.
""".strip()


WEATHER_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "도시의 현재 기온, 체감온도, 습도, 강수량, 날씨 코드, 풍속을 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "도시 이름"},
                "latitude": {"type": "number", "description": "도시 중심의 위도"},
                "longitude": {"type": "number", "description": "도시 중심의 경도"},
            },
            "required": ["city", "latitude", "longitude"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def get_weather(city: str, latitude: float, longitude: float) -> str:
    """Open-Meteo에서 지정한 좌표의 현재 날씨를 JSON 문자열로 반환한다."""
    if not -90 <= latitude <= 90:
        raise ValueError("위도는 -90에서 90 사이여야 합니다.")
    if not -180 <= longitude <= 180:
        raise ValueError("경도는 -180에서 180 사이여야 합니다.")

    params = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,weather_code,wind_speed_10m,is_day"
            ),
            "timezone": "auto",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("현재 날씨 서비스에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.") from exc

    current = data.get("current")
    if not current:
        raise RuntimeError("날씨 서비스 응답에 현재 날씨 정보가 없습니다.")

    result = {
        "city": city,
        "timezone": data.get("timezone"),
        **current,
        "units": data.get("current_units", {}),
    }
    return json.dumps(result, ensure_ascii=False)


class WeatherPlanner:
    """하나의 사용자 세션에 대한 대화 상태와 토큰 사용량을 관리한다."""

    def __init__(self, api_key: str, safety_identifier: str):
        self.client = OpenAI(api_key=api_key, max_retries=2, timeout=30.0)
        self.safety_identifier = safety_identifier
        self.previous_response_id: str | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.last_plan: DailyPlan | None = None

    def _moderate(self, text: str) -> None:
        result = self.client.moderations.create(
            model=MODERATION_MODEL,
            input=text,
        ).results[0]
        if result.flagged:
            raise ValueError("안전 검사를 통과하지 못한 요청입니다. 표현을 바꿔 다시 시도해 주세요.")

    def _record_usage(self, response) -> None:
        if response.usage is None:
            return
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        details = response.usage.input_tokens_details
        self.total_cached_tokens += getattr(details, "cached_tokens", 0) or 0

    def ask(self, user_request: str) -> DailyPlan:
        """사용자 요청을 검사하고 현재 날씨에 근거한 구조화 계획을 생성한다."""
        user_request = user_request.strip()
        if not user_request:
            raise ValueError("계획 요청을 입력해 주세요.")
        self._moderate(user_request)

        tool_response = self.client.responses.create(
            model=API_MODEL,
            instructions=PLANNER_INSTRUCTIONS,
            input=user_request,
            previous_response_id=self.previous_response_id,
            tools=WEATHER_TOOLS,
            tool_choice={"type": "function", "name": "get_weather"},
            parallel_tool_calls=False,
            prompt_cache_key="weather-planner-v1",
            safety_identifier=self.safety_identifier,
            metadata={"project": "weather-daily-planner", "step": "weather"},
        )
        self._record_usage(tool_response)

        tool_outputs = []
        for item in tool_response.output:
            if item.type != "function_call":
                continue
            if item.name != "get_weather":
                raise ValueError(f"지원하지 않는 도구입니다: {item.name}")
            arguments = json.loads(item.arguments)
            weather_json = get_weather(**arguments)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": weather_json,
                }
            )

        if not tool_outputs:
            raise RuntimeError("모델이 날씨 함수를 호출하지 않았습니다.")

        final_response = self.client.responses.parse(
            model=API_MODEL,
            instructions=PLANNER_INSTRUCTIONS,
            previous_response_id=tool_response.id,
            input=tool_outputs,
            tools=WEATHER_TOOLS,
            tool_choice="none",
            text_format=DailyPlan,
            prompt_cache_key="weather-planner-v1",
            safety_identifier=self.safety_identifier,
            metadata={"project": "weather-daily-planner", "step": "plan"},
        )
        self._record_usage(final_response)

        plan = final_response.output_parsed
        if plan is None:
            raise RuntimeError("모델 응답을 DailyPlan 형식으로 변환하지 못했습니다.")

        self.previous_response_id = final_response.id
        self.last_plan = plan
        return plan

    def stream_briefing(
        self,
        on_delta: Callable[[str], None] | None = None,
        request: str = "방금 계획을 친근한 한국어 브리핑으로 설명해줘.",
    ) -> str:
        """마지막 계획을 자연어로 스트리밍하고 완성된 문자열을 반환한다."""
        if self.previous_response_id is None:
            raise RuntimeError("먼저 계획을 만들어 주세요.")
        self._moderate(request)

        chunks: list[str] = []
        with self.client.responses.stream(
            model=API_MODEL,
            instructions="이전 계획을 바탕으로 간결하고 친근한 한국어 브리핑을 작성하세요.",
            input=request,
            previous_response_id=self.previous_response_id,
            prompt_cache_key="weather-planner-v1",
            safety_identifier=self.safety_identifier,
        ) as stream:
            for event in stream:
                if event.type != "response.output_text.delta":
                    continue
                chunks.append(event.delta)
                if on_delta is not None:
                    on_delta(event.delta)
            response = stream.get_final_response()

        self._record_usage(response)
        self.previous_response_id = response.id
        return "".join(chunks) or response.output_text

    def create_speech(self, text: str) -> bytes:
        """텍스트 브리핑을 MP3 바이트로 변환한다."""
        if not text.strip():
            raise ValueError("음성으로 변환할 브리핑이 없습니다.")
        speech = self.client.audio.speech.create(
            model=SPEECH_MODEL,
            voice="alloy",
            input=text,
            response_format="mp3",
        )
        return speech.content

    def usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    def reset(self) -> None:
        """대화 연결과 마지막 계획을 초기화한다. 누적 토큰은 유지한다."""
        self.previous_response_id = None
        self.last_plan = None
