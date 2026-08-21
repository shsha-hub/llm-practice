# LLM Study Mate

Day02의 청킹, 임베딩, 벡터 검색, Chroma와 RAG를 활용한 Streamlit 개념 학습 챗봇입니다. 학습 자료 안에서만 수준별 설명을 만들고, 주관식 퀴즈를 생성해 답안의 핵심 의미를 평가합니다.

## 주요 기능

- 입문·일반·심화 수준별 근거 기반 설명
- 자유 질문과 주제별 학습
- 학습 자료 기반 주관식 문제 생성
- 정답·부분 정답·오답 판정과 구체적인 피드백
- 답변과 채점에 사용된 원문 조각 확인

## 실행 방법

프로젝트 폴더로 이동한 다음 환경 변수를 준비합니다.

```bash
cp .env.example .env
```

`.env` 파일의 `OPENAI_API_KEY`를 실제 키로 변경합니다. 필요한 패키지가 없다면 설치합니다.

```bash
uv pip install -r requirements.txt
```

학습 자료를 청킹하고 Chroma에 저장합니다.

```bash
python ingest.py
```

그다음 앱을 실행합니다.

```bash
streamlit run app.py
```

학습 자료를 수정한 뒤 색인을 다시 만들 때는 다음 명령을 사용합니다.

```bash
python ingest.py --rebuild
```

## 구성

```text
llm-study-mate/
├── data/llm-study-guide.md  # 수업용 학습 자료
├── app.py                   # Streamlit 화면과 학습 흐름
├── ingest.py                # 청킹 및 Chroma 색인
├── study_core.py            # 검색·설명·퀴즈·채점 로직
└── requirements.txt
```

## 확인할 테스트 질문

- LLM은 일반 프로그램과 무엇이 다른가요?
- 임베딩은 왜 검색에 사용하나요?
- 청크를 너무 작게 나누면 어떤 문제가 생기나요?
- RAG는 환각을 완전히 없앨 수 있나요?
- 오늘 서울 날씨는 어떤가요? (학습 자료 범위 밖 질문)

자동 생성된 설명과 채점은 학습 보조용입니다. 중요한 내용은 화면에 표시되는 원문 근거와 함께 확인하세요.
