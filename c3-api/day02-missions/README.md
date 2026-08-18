# 지진 조회·트리아지봇

USGS 실시간 지진 데이터를 조회(**도구 호출**)하고, 위험도를 정해진 형식의 데이터로 판정(**구조화 출력**)하는 챗봇입니다.

- `quake_triage.py` — 터미널 버전
- `quake_app.py` — Streamlit 웹 버전

---

## 실행 방법

```bash
pip install openai pydantic streamlit
export OPENAI_API_KEY=sk-...

# 터미널 버전
python quake_triage.py

# 웹 버전
streamlit run quake_app.py --server.headless true
```

`--server.headless true`는 GUI가 없는 환경(서버·컨테이너 등)에서 브라우저 자동 실행 시도를 막기 위한 옵션입니다. 실행 후 안내되는 `http://localhost:8501`을 직접 브라우저로 열면 됩니다.

---

## 무엇을 할 수 있나

USGS Earthquake API(키 불필요)를 조회해서, **지금 시점부터 원하는 만큼 거슬러 올라간 기간** 안의 지진을 규모 조건과 함께 찾아줍니다.

| 가능한 질문 예시 | 안 되는 것 |
|---|---|
| "최근 3시간 동안 규모 5 이상 지진 있어?" | "2011년 동일본대지진 규모 알려줘" (특정 과거 날짜) |
| "최근 1주일간 규모 4 이상 지진 알려줘" | "역대 최대 지진이 뭐야?" (절대 시점 조회) |
| "지금 위험한 지진 있는지 판단해줘" | |

→ `hours` 파라미터는 **상한이 없어서** "최근 1개월"처럼 큰 기간도 요청 가능하지만, 항상 **지금을 기준으로 한 상대 시간**만 다룹니다.

---

## 동작 구조

```
사용자 질문
   │
   ▼
① 도구 호출 왕복 (최대 5바퀴)
   get_recent_earthquakes(min_magnitude, hours) 로 USGS 조회
   → 결과를 role: "tool" 로 배열에 재투입
   │
   ▼
② 구조화 출력 (Pydantic)
   EarthquakeTriage { risk_level, summary, biggest_quake, quake_list }
   → 위험도·요약·최대 지진·전체 목록을 정해진 형식의 데이터로 강제
   │
   ▼
화면에 위험도 배지(🟢🟡🔴) + 요약 + 전체 목록 표시
```

- **도구 호출**: 모델은 "USGS에 물어봐 달라"는 요청서만 만들고, 실제 HTTP 요청은 내 코드(`urllib`)가 실행합니다.
- **구조화 출력**: `summary`처럼 자유 서술 필드는 모델 재량이 크므로, 반드시 보여줘야 하는 정보(지진 목록)는 `quake_list: list[str]`처럼 전용 필드로 스키마에 못박아 둡니다.

---

## 파일 구성

```
quake_triage.py   # 도구 정의 + 트리아지 로직 (터미널에서 1회 실행)
quake_app.py      # 위 로직을 Streamlit 채팅 UI로 감싼 버전
```

---

## 알려진 제약 · 주의사항

- USGS 응답 상위 10건까지만 사용 (`features[:10]`)
- 조회 기간이 길어 지진이 많이 잡히면 `quake_list`가 길어져 `max_completion_tokens`(900)를 넘길 수 있음 → `LengthFinishReasonError` 발생 시 화면에 안내 메시지 표시, 기간을 좁혀서 재질문 필요
- `st.session_state.messages`에는 dict(사용자·도구 결과)와 `ChatCompletionMessage` 객체(도구 호출 중간 응답)가 섞여 저장됨 → 화면 렌더링 시 `isinstance` 체크로 안전하게 처리
