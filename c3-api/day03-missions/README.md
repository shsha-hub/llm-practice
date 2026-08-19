# 날씨 기반 맞춤형 하루 계획 도우미

현재 날씨와 사용자의 지역·가능 시간·선호 활동을 결합해 현실적인 하루 계획을 만들어 주는 Jupyter Notebook 및 Streamlit 미니 프로젝트입니다.

단순히 모델에게 날씨를 물어보는 것이 아니라, 모델이 `get_weather` 함수 호출을 요청하면 파이썬 코드가 Open-Meteo API에서 실제 날씨를 조회하고, 그 결과를 다시 모델에 전달해 `DailyPlan` 객체를 생성합니다. 이후에는 이전 계획을 기억한 수정 요청, 스트리밍 브리핑, 토큰 사용량 확인, 음성 파일 생성까지 실습할 수 있습니다.

Jupyter 실습은 [mission.ipynb](./mission.ipynb), 웹 서비스는 [app.py](./app.py)에서 실행합니다.

---

## 주요 기능

| 기능 | 프로젝트에서 하는 일 |
|---|---|
| Responses API | 함수 호출 요청과 최종 계획 생성 |
| Function Calling | 모델이 `get_weather` 실행에 필요한 도시와 좌표 결정 |
| Open-Meteo API | 실제 현재 날씨 조회 |
| Structured Outputs | 결과를 `DailyPlan` Pydantic 객체로 반환 |
| `previous_response_id` | 이전 계획을 기억한 후속 수정 요청 |
| Streaming | 완성된 계획을 자연어 브리핑으로 실시간 출력 |
| Moderation | 사용자 입력을 계획 생성 전에 안전 검사 |
| Usage / Prompt Cache | 입력·출력·캐시 토큰 누적 확인 |
| TTS | 브리핑을 MP3 음성 파일로 저장 |
| Metadata / Safety Identifier | 요청 용도와 최종 사용자 식별 정보 전달 |

`gpt-5.6-luna`는 Responses API, Streaming, Function Calling, Structured Outputs를 지원합니다. 정확한 지원 범위는 [GPT-5.6 Luna 공식 문서](https://developers.openai.com/api/docs/models/gpt-5.6-luna)에서 확인할 수 있습니다.

---

## 프로젝트 파일

```text
day03-missions/
├── app.py                    # Streamlit 채팅 UI 및 세션 관리
├── weather_planner.py        # 공용 API·도메인 로직
├── mission.ipynb             # 프로젝트 구현 및 실행 노트북
├── requirements.txt          # Streamlit 서비스 실행 의존성
├── README.md                 # 프로젝트 설명서
├── daily_plan_seoul.mp3      # 서울 계획 음성 브리핑 실행 결과
└── daily_plan_busan.mp3      # 부산 계획 음성 브리핑 실행 결과
```

MP3 파일은 노트북의 TTS 셀을 실행한 결과물입니다. 같은 파일명으로 다시 실행하면 기존 파일을 덮어씁니다.

---

## 실행 환경

이 프로젝트는 다음 환경을 기준으로 작성되었습니다.

- Python 3.12
- OpenAI Python SDK 3.x
- Pydantic 2.x
- Jupyter Notebook, VS Code Notebook 또는 Streamlit
- 인터넷 연결
- OpenAI API 키

필요한 패키지가 없다면 가상환경에서 설치합니다.

```bash
pip install -r day03-missions/requirements.txt
```

노트북 커널도 새로 구성해야 한다면 `ipykernel`을 추가로 설치합니다.

```bash
pip install ipykernel
```

Open-Meteo 호출에는 별도 API 키가 필요하지 않지만, OpenAI Responses·Moderation·TTS API 호출에는 OpenAI API 키와 사용 가능한 계정 한도가 필요합니다.

### API 키 설정

Linux 또는 macOS에서는 터미널에서 다음과 같이 설정합니다.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

API 키는 노트북 코드나 Git 저장소에 직접 작성하지 않는 것이 좋습니다. 첫 번째 코드 셀은 키의 존재 여부만 확인하고 실제 키 값은 출력하지 않습니다.

---

## 실행 방법

### Streamlit 웹 서비스

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
streamlit run day03-missions/app.py
```

GUI가 없는 서버나 컨테이너에서는 다음처럼 실행할 수 있습니다.

```bash
streamlit run day03-missions/app.py --server.headless true
```

터미널에 표시되는 주소를 브라우저에서 열면 됩니다. 기본 로컬 주소는 일반적으로 `http://localhost:8501`입니다.

웹 앱에서 제공하는 기능은 다음과 같습니다.

- 채팅 입력으로 첫 계획 및 후속 수정 요청
- 현재 날씨와 활동을 카드 형태로 표시
- 사용자 브라우저 세션별 `WeatherPlanner`와 대화 기록 분리
- 선택 가능한 AI 자연어 스트리밍 브리핑
- Responses 입력·출력·캐시 토큰 표시
- 새 대화 시작 버튼
- 최신 계획의 AI 음성 생성·재생·MP3 다운로드
- API 연결·인증·한도 오류 안내

배포 환경에서는 `OPENAI_API_KEY` 환경변수 또는 Streamlit secrets를 사용할 수 있습니다. `.streamlit/secrets.toml`을 사용할 경우 다음과 같이 설정하되 파일을 Git에 커밋하지 마세요.

```toml
OPENAI_API_KEY = "YOUR_API_KEY"
```

### Jupyter Notebook

1. [mission.ipynb](./mission.ipynb)을 엽니다.
2. 프로젝트의 `.venv` 또는 필요한 패키지가 설치된 Python 커널을 선택합니다.
3. `OPENAI_API_KEY`가 설정되어 있는지 확인합니다.
4. 위 셀부터 아래 셀까지 순서대로 실행합니다.
5. 첫 요청 문장을 원하는 도시·시간·활동으로 수정해 봅니다.
6. 음성이 필요하지 않다면 TTS 셀은 실행하지 않아도 됩니다.

기본 예시는 다음 요청으로 계획을 생성합니다.

```python
planner = WeatherPlanner()
plan = planner.ask(
    "부산에 살고 오늘 저녁 9시부터 2시간 동안 가볍게 운동하고 싶어."
)
display(plan)
```

후속 요청에서는 도시와 시간을 반복하지 않아도 이전 계획의 문맥을 사용할 수 있습니다.

```python
revised_plan = planner.ask(
    "야외 활동은 빼고, 같은 시간 안에서 실내 계획으로 바꿔줘."
)
display(revised_plan)
```

---

## 전체 동작 구조

```text
사용자 요청
   │
   ▼
① Moderation API
   └─ 위험한 입력이면 ValueError로 중단
   │
   ▼
② Responses API: 함수 호출 요청
   ├─ instructions: 계획 도우미의 행동 규칙
   ├─ tools: get_weather JSON Schema
   └─ tool_choice: get_weather를 반드시 한 번 호출
   │
   ▼
③ 파이썬 get_weather 함수 실행
   └─ Open-Meteo에서 현재 날씨 조회
   │
   ▼
④ function_call_output 전달
   └─ call_id로 함수 호출과 실행 결과 연결
   │
   ▼
⑤ Responses API: 구조화된 최종 계획 생성
   ├─ text_format=DailyPlan
   └─ output_parsed에서 Pydantic 객체 획득
   │
   ├───────────────┬────────────────┐
   ▼               ▼                ▼
후속 수정 요청   스트리밍 설명    TTS 음성 파일
```

첫 번째 `planner.ask()` 호출에는 일반적으로 다음 네 번의 외부 요청이 포함됩니다.

1. OpenAI Moderation 요청
2. OpenAI Responses 함수 호출 요청
3. Open-Meteo 날씨 요청
4. OpenAI Responses 구조화 출력 요청

따라서 `ask()`는 로컬 함수처럼 보이지만 네트워크·API 사용량·응답 지연이 발생할 수 있습니다.

### Streamlit 구성

웹 서비스는 화면과 API 로직을 분리했습니다.

```text
app.py
├── Streamlit 채팅 UI
├── session_state 대화·오디오 관리
├── 계획 카드 렌더링
└── 사용자 친화적 오류 메시지
        │
        ▼
weather_planner.py
├── Activity / DailyPlan
├── Open-Meteo 조회
├── Function Calling 왕복
├── previous_response_id
├── Responses 스트리밍
├── 토큰 집계
└── TTS 생성
```

`WeatherPlanner`를 전역 캐시로 공유하지 않고 `st.session_state`에 보관하므로, 브라우저 사용자 세션마다 대화 상태와 응답 ID가 따로 유지됩니다. 화면을 새로고침해 같은 Streamlit 세션이 유지되는 동안에는 대화를 이어갈 수 있지만, 세션이 종료되거나 서버가 재시작되면 앱의 화면 기록은 사라집니다.

---

## 핵심 구현 설명

### 1. 클라이언트 설정

```python
client = OpenAI(max_retries=2, timeout=30.0)
API_MODEL = "gpt-5.6-luna"
```

- `max_retries=2`: 재시도 가능한 일시적 API 오류가 발생하면 최대 두 번 재시도합니다.
- `timeout=30.0`: 한 요청을 최대 30초 동안 기다립니다.
- `API_MODEL`: 프로젝트의 Responses 요청에 공통으로 사용하는 모델입니다.

재시도는 OpenAI SDK 요청에만 적용됩니다. `urllib.request.urlopen(..., timeout=10)`으로 수행하는 Open-Meteo 호출에는 별도 재시도 로직이 없습니다.

### 2. 구조화된 출력

프로젝트는 두 개의 Pydantic 모델을 사용합니다.

```text
DailyPlan
├── city: str
├── weather_summary: str
├── activities: list[Activity]
│   ├── time: str
│   ├── title: str
│   ├── location: str
│   ├── reason: str
│   └── preparation: list[str]
└── caution: str
```

`Activity`는 하나의 활동을, `DailyPlan`은 전체 계획을 표현합니다. `Field(description=...)`은 모델에게 각 필드에 어떤 내용을 채워야 하는지 알려 줍니다.

최종 요청에서 다음과 같이 사용합니다.

```python
final_response = client.responses.parse(
    ...,
    text_format=DailyPlan,
)
plan = final_response.output_parsed
```

`output_parsed`는 자유 형식 문자열이 아니라 검증된 `DailyPlan` 객체입니다. 다만 스키마를 지킨다는 것이 계획 내용의 사실성이나 적절성을 완전히 보장한다는 뜻은 아닙니다. 날씨 정보의 정확성, 프롬프트, 모델 판단도 함께 검토해야 합니다.

### 3. Open-Meteo 날씨 함수

```python
def get_weather(city: str, latitude: float, longitude: float) -> str:
```

이 함수는 다음 현재 날씨 항목을 조회합니다.

- 기온 `temperature_2m`
- 체감온도 `apparent_temperature`
- 습도 `relative_humidity_2m`
- 강수량 `precipitation`
- WMO 날씨 코드 `weather_code`
- 풍속 `wind_speed_10m`
- 낮/밤 여부 `is_day`

함수는 Open-Meteo 응답 중 현재 날씨와 단위를 추려 JSON 문자열로 반환합니다. Function Calling 결과의 `output`에는 문자열을 전달할 수 있으므로, `json.dumps(..., ensure_ascii=False)`를 사용해 한글 도시명도 읽기 쉬운 형태로 유지합니다.

현재 구현에서는 모델이 도시의 위도와 경도를 생성합니다. 일반적인 도시에서는 편리하지만, 동명이인 도시나 세부 지역에서는 좌표가 부정확할 수 있습니다. 실제 서비스로 확장한다면 별도의 지오코딩 API로 도시를 좌표로 변환하는 편이 안전합니다.

### 4. Function Tool 정의

`WEATHER_TOOLS`는 파이썬 함수를 직접 전달하는 값이 아니라 모델이 이해할 수 있는 함수 명세입니다.

```python
{
    "type": "function",
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {...},
        "required": ["city", "latitude", "longitude"],
        "additionalProperties": False,
    },
    "strict": True,
}
```

- `required`: 세 인자를 모두 필수로 지정합니다.
- `additionalProperties=False`: 정의되지 않은 인자를 만들지 못하게 합니다.
- `strict=True`: 함수 인자가 JSON Schema를 엄격하게 따르도록 합니다.

중요한 점은 모델이 `get_weather()`를 직접 실행하지 않는다는 것입니다. 모델은 함수 호출 요청만 만들고, 아래 코드가 실제 함수를 실행합니다.

```python
arguments = json.loads(item.arguments)
weather_json = get_weather(**arguments)
```

함수 호출의 전체 개념과 엄격한 스키마 조건은 [OpenAI Function Calling 문서](https://developers.openai.com/api/docs/guides/function-calling)를 참고할 수 있습니다.

### 5. 함수 결과와 `call_id`

모델의 함수 호출 요청에는 `call_id`가 들어 있습니다. 실행 결과를 반환할 때 같은 ID를 사용해야 합니다.

```python
tool_outputs.append({
    "type": "function_call_output",
    "call_id": item.call_id,
    "output": weather_json,
})
```

이를 통해 모델은 여러 출력 항목이 있어도 어떤 실행 결과가 어떤 함수 호출에 대응하는지 구분할 수 있습니다.

### 6. 계획 도우미 지침

`PLANNER_INSTRUCTIONS`에는 모델의 역할과 경계가 들어 있습니다.

- 사용자의 지역·시간·선호 활동 반영
- 도구 결과만 현재 날씨의 근거로 사용
- 강수·폭염·한파·강풍 시 무리한 야외 활동 회피
- 사용자가 밝히지 않은 건강 상태나 취향을 임의로 단정하지 않음
- `DailyPlan` 스키마 준수

`previous_response_id`를 사용하는 후속 요청에서도 동작 규칙이 분명하도록 각 Responses 요청에 `instructions`를 다시 전달합니다.

### 7. `WeatherPlanner` 상태

`WeatherPlanner` 객체는 다음 상태를 보관합니다.

| 속성 | 의미 |
|---|---|
| `previous_response_id` | 다음 요청에 연결할 마지막 응답 ID |
| `total_input_tokens` | 누적 Responses 입력 토큰 |
| `total_output_tokens` | 누적 Responses 출력 토큰 |
| `total_cached_tokens` | 누적 캐시 입력 토큰 |
| `last_plan` | 마지막 `DailyPlan` 객체 |

`reset()`은 대화 ID와 마지막 계획만 초기화합니다.

```python
planner.reset()
```

토큰 누적값은 초기화하지 않으므로, 새 대화를 시작한 뒤에도 `usage()`는 해당 `WeatherPlanner` 객체가 사용한 전체 토큰을 보여 줍니다.

### 8. `ask()` 메서드

`ask()`는 프로젝트의 핵심 오케스트레이션 메서드입니다.

```python
plan = planner.ask("부산에서 오늘 저녁 실내 운동 계획을 만들어줘.")
```

내부 실행 순서는 다음과 같습니다.

1. `_moderate()`로 사용자 입력 검사
2. `responses.create()`로 `get_weather` 호출 요청 생성
3. 모델이 반환한 함수 인자를 `json.loads()`로 파싱
4. 파이썬 `get_weather()` 실행
5. 실행 결과를 `function_call_output`으로 전달
6. `responses.parse()`로 `DailyPlan` 생성
7. 응답 ID, 마지막 계획, 토큰 사용량 저장

```python
tool_choice={"type": "function", "name": "get_weather"}
parallel_tool_calls=False
```

이 설정은 계획을 만들 때 항상 날씨를 한 번 조회하도록 하고, 한 요청에서 여러 날씨 함수가 동시에 호출되는 것을 막습니다.

최종 계획 요청에서는 이미 날씨를 얻었으므로 다음과 같이 추가 함수 호출을 막습니다.

```python
tool_choice="none"
```

### 9. 대화 상태와 후속 요청

첫 요청이 끝나면 최종 응답 ID를 저장합니다.

```python
self.previous_response_id = final_response.id
```

다음 `ask()` 호출에서는 이 값을 전달합니다.

```python
previous_response_id=self.previous_response_id
```

따라서 사용자는 도시와 시간을 모두 다시 입력하지 않고도 다음처럼 요청할 수 있습니다.

```text
"야외 활동은 빼고, 같은 시간 안에서 실내 계획으로 바꿔줘."
```

도구 호출과 최종 구조화 출력 사이에서도 `previous_response_id=tool_response.id`를 사용합니다. 이 연결 덕분에 모델이 앞선 함수 호출 문맥과 `function_call_output`을 함께 해석할 수 있습니다. 자세한 내용은 [OpenAI Conversation State 문서](https://developers.openai.com/api/docs/guides/conversation-state)를 참고하세요.

### 10. Moderation

각 사용자 요청과 스트리밍 브리핑 요청은 먼저 `omni-moderation-latest`로 검사합니다.

```python
result = client.moderations.create(
    model="omni-moderation-latest",
    input=text,
).results[0]
```

`result.flagged`가 참이면 `ValueError`를 발생시키고 처리를 중단합니다. 현재 구현은 입력만 검사하며, Open-Meteo 결과나 모델의 최종 출력에 대한 별도 후처리 검사는 수행하지 않습니다.

### 11. 스트리밍 브리핑

```python
briefing_text = planner.stream_briefing()
```

`stream_briefing()`은 마지막 구조화 계획을 사람이 읽기 편한 자연어로 다시 설명합니다.

```python
if event.type == "response.output_text.delta":
    print(event.delta, end="", flush=True)
```

완성된 응답을 한꺼번에 기다리지 않고 텍스트 조각이 도착할 때마다 출력합니다. 스트림 종료 후 `get_final_response()`로 완성된 응답을 얻어 토큰과 응답 ID를 저장합니다. 이벤트 방식은 [OpenAI Streaming 문서](https://developers.openai.com/api/docs/guides/streaming-responses)를 참고하세요.

### 12. 토큰 및 프롬프트 캐시

`_record_usage()`는 Responses 요청의 사용량을 누적합니다.

```python
planner.usage()
```

반환 예시는 다음과 같습니다.

```python
{
    "input_tokens": 2100,
    "output_tokens": 500,
    "cached_tokens": 900,
    "total_tokens": 2600,
}
```

`total_tokens`는 입력과 출력의 합이며, `cached_tokens`는 입력 토큰에 포함된 부분 집계입니다. 따라서 `total_tokens`에 `cached_tokens`를 다시 더하면 안 됩니다.

`prompt_cache_key="weather-planner-v1"`는 관련 요청에 일관된 캐시 키를 제공합니다. 키를 설정했다고 항상 캐시 적중이 발생하는 것은 아니며, 실제 적용 여부는 `cached_tokens`로 확인해야 합니다.

Moderation API와 TTS 요청 사용량은 현재 `WeatherPlanner.usage()`에 포함되지 않습니다.

### 13. Metadata와 Safety Identifier

각 계획 생성 요청에는 다음 값이 포함됩니다.

```python
metadata={"project": "weather-daily-planner", "step": "weather"}
safety_identifier="mission-notebook-user"
```

- `metadata`: 프로젝트와 처리 단계를 구분하기 위한 태그입니다.
- `safety_identifier`: 최종 사용자를 안정적으로 구분하기 위한 값입니다.

노트북의 `safety_identifier`는 수업용 고정 문자열입니다. Streamlit 앱은 브라우저 세션 ID를 SHA-256으로 해시해 세션별 식별자를 만듭니다. 실제 로그인 사용자가 있는 서비스에서는 이메일이나 이름 같은 개인정보를 직접 보내지 말고, 사용자별로 안정적이면서 개인정보를 노출하지 않는 해시 형태의 값을 사용하는 것이 좋습니다.

### 14. TTS 음성 브리핑

```python
audio_path = save_voice_briefing(briefing_text)
display(Audio(filename=str(audio_path)))
```

스트리밍에서 완성한 `briefing_text`를 음성으로 변환하고 `daily_plan_busan.mp3`로 저장합니다. 저장 후 Jupyter의 `Audio` 위젯으로 바로 재생할 수 있습니다.

현재 노트북의 `gpt-4o-mini-tts`는 강의에서 사용한 모델을 그대로 유지한 것입니다. 모델 제공 상태와 권장 음성 모델은 변경될 수 있으므로 실행 오류가 발생하면 [OpenAI 모델 목록](https://developers.openai.com/api/docs/models)과 음성 관련 공식 문서를 확인하세요.

Streamlit 앱은 현재 제공되는 Speech API 모델인 `tts-1`을 사용하고, 생성된 MP3를 서버 파일로 저장하지 않고 브라우저 세션 메모리에 보관합니다. 사용자는 화면에서 재생하거나 다운로드할 수 있으며, 앱에는 AI 생성 음성임을 표시합니다.

---

## 셀별 실행 가이드

| 셀 | 종류 | 역할 | 외부 요청 |
|---:|---|---|---|
| 1 | Markdown | 프로젝트 소개와 전체 흐름 | 없음 |
| 2 | Code | 라이브러리, API 키, OpenAI 클라이언트 설정 | 없음 |
| 3 | Markdown | 강의 기능과 프로젝트 구현 연결 | 없음 |
| 4 | Markdown | 구조화 출력 섹션 제목 | 없음 |
| 5 | Code | `Activity`, `DailyPlan` 스키마 정의 | 없음 |
| 6 | Markdown | 날씨 Function Tool 설명 | 없음 |
| 7 | Code | `get_weather()`와 `WEATHER_TOOLS` 정의 | 정의만 할 때는 없음 |
| 8 | Markdown | `WeatherPlanner` 역할 설명 | 없음 |
| 9 | Code | 지침과 `WeatherPlanner` 클래스 정의 | 정의만 할 때는 없음 |
| 10 | Markdown | 실행 방법 안내 | 없음 |
| 11 | Code | 부산 기준 첫 계획 생성 | Moderation, Responses 2회, Open-Meteo |
| 12 | Code | 이전 문맥을 사용한 실내 계획 수정 | Moderation, Responses 2회, Open-Meteo |
| 13 | Code | 자연어 스트리밍 및 토큰 확인 | Moderation, Responses Streaming |
| 14 | Markdown | TTS 선택 기능 안내 | 없음 |
| 15 | Code | 브리핑 MP3 생성 및 재생 | OpenAI TTS |
| 16 | Markdown | 추가 실험 아이디어와 공식 문서 링크 | 없음 |

노트북 커널을 재시작하면 메모리에 정의한 클래스와 `planner` 객체가 사라지므로 셀 2부터 다시 순서대로 실행해야 합니다.

---

## 출력 예시 형태

실제 내용은 날씨와 모델 응답에 따라 달라지지만 구조는 다음과 같습니다.

```python
DailyPlan(
    city="부산",
    weather_summary="현재 기온과 강수·풍속을 요약한 내용",
    activities=[
        Activity(
            time="21:00",
            title="실내 유산소 운동",
            location="가까운 피트니스 센터",
            reason="현재 날씨와 사용자의 가벼운 운동 선호를 반영",
            preparation=["운동복", "실내 운동화", "물"]
        )
    ],
    caution="운동 강도와 이동 시 주의사항"
)
```

이 예시는 형식을 설명하기 위한 것이며 실제 현재 날씨를 나타내지 않습니다.

---

## 알려진 제약과 주의사항

- 현재 날씨만 조회하며 시간대별 예보나 미래 날씨는 사용하지 않습니다.
- 도시 좌표를 모델이 생성하므로 잘 알려지지 않은 지역이나 동명이인 도시에서 좌표가 틀릴 수 있습니다.
- 후속 요청에서도 `get_weather`를 강제로 호출하므로 매번 현재 날씨를 다시 조회합니다.
- Open-Meteo 호출에는 재시도 처리가 없어서 일시적인 네트워크 오류가 바로 예외로 전달됩니다.
- `tool_choice`로 함수 호출을 강제하므로 도시를 전혀 알 수 없는 요청도 모델이 좌표를 추정하려 할 수 있습니다.
- 구조화 출력은 데이터 모양을 보장하지만 추천의 현실성·안전성·의학적 적합성을 보장하지 않습니다.
- Moderation은 사용자 입력만 검사하며 모델 출력에 대한 별도 검사는 없습니다.
- `safety_identifier`는 데모용 고정값이므로 다중 사용자 서비스에 그대로 사용하면 안 됩니다.
- `reset()`은 대화 상태만 초기화하고 누적 토큰은 초기화하지 않습니다.
- API 호출과 TTS 생성에는 계정 상태와 사용량에 따라 비용이 발생할 수 있습니다.
- MP3 저장 경로는 노트북 커널의 현재 작업 디렉터리를 기준으로 합니다.

이 프로젝트는 학습용 예제입니다. 폭염·한파·대기질·건강 상태처럼 안전에 영향을 줄 수 있는 요소가 중요한 상황에서는 전문 기상 정보와 개인 상태를 별도로 확인해야 합니다.

---

## 확장 아이디어

### 난이도 하

- `reset_usage()` 메서드를 추가해 토큰 통계도 초기화
- 운동 강도, 예산, 이동 거리 필드를 `DailyPlan`에 추가
- 날씨 코드를 맑음·비·눈 같은 한글 설명으로 변환
- Open-Meteo 오류를 `try/except`로 처리해 사용자 친화적 메시지 출력

### 난이도 중

- Open-Meteo 시간대별 예보를 사용해 활동 시간의 날씨를 기준으로 계획 생성
- 지오코딩 API를 추가해 도시명에서 정확한 좌표 조회
- Gradio 또는 Streamlit으로 웹 채팅 UI 제작
- 대화별 토큰, 응답 시간, 첫 토큰 도착 시간을 표로 기록
- 사용자별 개인정보 비노출 `safety_identifier` 생성

### 난이도 상

- 여러 날짜의 계획을 비교하는 주간 플래너
- 공기질·자외선·대중교통·장소 검색 도구 추가
- 실패한 외부 요청에 지수 백오프 재시도 적용
- 대표 요청과 기대 조건을 준비해 모델·프롬프트별 평가 자동화
- 저장하지 않는 대화 정책이나 별도 대화 저장소 설계

---

## 공식 참고 문서

- [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [Function Calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Conversation State](https://developers.openai.com/api/docs/guides/conversation-state)
- [Streaming Responses](https://developers.openai.com/api/docs/guides/streaming-responses)
- [Prompt Caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Moderation](https://developers.openai.com/api/docs/guides/moderation)
- [Text-to-Speech](https://developers.openai.com/api/docs/guides/text-to-speech)

OpenAI API의 모델·매개변수·지원 범위는 변경될 수 있으므로, 실행 시점에는 공식 OpenAI 문서를 함께 확인하는 것이 좋습니다.
