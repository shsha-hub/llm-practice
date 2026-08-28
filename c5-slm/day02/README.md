# Day 02 미니 과제: 내 PC에 맞는 로컬 LLM 선택기

같은 프롬프트를 여러 Ollama 모델에 보내 양자화 등급별 파일 크기, VRAM,
응답 시간, 생성 속도를 비교하는 Streamlit 앱입니다. 큰 모델 실행이 실패하면
더 작은 모델로 전환하는 fallback도 직접 실험할 수 있습니다.

## 실행

먼저 별도 터미널에서 Ollama를 실행하고 비교할 모델을 준비합니다.

```bash
ollama serve
ollama list
```

프로젝트 가상환경에서 앱을 실행합니다.

```bash
cd /home/student/llm-practice/c5-slm
.venv/bin/streamlit run day02/streamlit_app.py
```

브라우저에서 모델을 2개 이상 선택하고 **벤치마크 실행**을 누릅니다.
가능하면 같은 base 모델의 Q2/Q4/Q8 버전을 비교하세요.

## 테스트

Ollama 서버 없이도 핵심 로직을 테스트할 수 있습니다.

```bash
.venv/bin/python -m unittest discover -s day02 -p 'test_*.py' -v
```

## 해석할 때 주의할 점

- `tok/s`는 Ollama가 보고한 실제 생성 구간을 기준으로 합니다.
- `시간(초)`에는 모델 로드 시간을 포함한 사용자가 체감하는 전체 시간이 들어갑니다.
- 균형 추천은 속도와 자원 사용량만 계산하며 답변의 정확도를 평가하지 않습니다.
- 모델 실행 뒤에는 다음 모델과 공정하게 비교할 수 있도록 메모리에서 내립니다.
