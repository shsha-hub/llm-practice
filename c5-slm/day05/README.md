# QLoRA 기반 멀티 생활 도우미

RTX 3050 Laptop(4GB VRAM)에서 Mi:dm 2B 모델 하나에 두 개의 LoRA 어댑터를
순차 학습하는 Day 05 과제입니다.

- `reply`: 곤란한 상황에서 바로 보낼 수 있는 2~3문장 답장
- `fridge`: 보유 재료를 우선 소진하는 15분 내외 레시피

원본 모델은 4비트로 한 번만 불러오며, 과제 결과는 작은 어댑터 파일로 저장합니다.
Mi:dm 기본 채팅 템플릿이 약 500토큰의 시스템 메시지를 자동 삽입하는 문제를 피하기
위해 짧은 Llama 형식의 `prompt/completion` 데이터로 학습합니다. 정답 부분에만 loss를
계산합니다.

## 복습용 Jupyter Notebook

스크립트는 재현과 제출용이고, 아래 노트북은 셀을 직접 실행하며 학습 과정을 확인하는
복습용입니다.

1. `notebooks/01_train_qlora_workbook.ipynb`
   - GPU와 VRAM 확인
   - 기본 chat template와 compact prompt 토큰 수 비교
   - 4-bit base 로드
   - LoRA 주입과 학습 파라미터 확인
   - smoke 1 step 및 full 3 epoch 학습
   - loss, validation loss, peak VRAM 확인
2. `notebooks/02_compare_adapters_workbook.ipynb`
   - base 하나에 두 adapter 등록
   - base / prompted base / adapter 직접 비교
   - 미학습 테스트셋 실행과 실패 사례 관찰

학습 노트북은 먼저 `TASK=\"reply\"`, `RUN_MODE=\"smoke\"`로 실행합니다. 성공하면
커널을 재시작하고 `RUN_MODE=\"full\"`로 바꿔 전체 학습합니다. 냉장고 task는 다시
커널을 재시작한 뒤 `TASK=\"fridge\"`로 같은 순서를 반복합니다.

## 실행 순서

프로젝트 루트에서 아래 명령을 실행합니다.

```bash
# 데이터와 포맷 검증
.venv/bin/python -m day05.scripts.validate

# 반드시 먼저 한 스텝만 실행해 VRAM 적합성 확인
.venv/bin/python -m day05.scripts.train --task reply --smoke

# 답장 어댑터 전체 학습
.venv/bin/python -m day05.scripts.train --task reply

# 냉장고 어댑터 전체 학습
.venv/bin/python -m day05.scripts.train --task fridge

# 미학습 테스트셋으로 base / prompt / adapter 비교
.venv/bin/python -m day05.scripts.compare --task reply
.venv/bin/python -m day05.scripts.compare --task fridge

# 저장된 비교 결과의 형식 준수율 계산
.venv/bin/python -m day05.scripts.evaluate --task reply
.venv/bin/python -m day05.scripts.evaluate --task fridge

# 두 어댑터를 실제 입력으로 체험하는 웹 앱
.venv/bin/streamlit run day05/app.py
```

Streamlit 앱은 base 모델을 GPU에 한 번만 올리고 `reply`, `fridge` 어댑터를 이름으로
등록해 선택한 task에 맞춰 전환합니다. 첫 생성은 모델 로딩 때문에 시간이 더 걸리며,
이후 생성에서는 Streamlit resource cache를 재사용합니다.

## 폴더 구조

```text
day05/
├── app.py                    # Streamlit 실행 진입점
├── life_assistant/           # 학습·추론 공용 설정과 생성 로직
├── scripts/                  # train, validate, compare, evaluate CLI
├── notebooks/                # 셀 단위 학습·비교 복습 워크북
├── data/                     # task별 train/val/test JSONL
├── outputs/                  # 로컬 어댑터와 비교 결과
├── results/                  # 학습 지표
├── README.md                 # 실행 안내
└── REPORT.md                 # 제출용 실험 보고서
```

학습 직전에는 `nvidia-smi`로 다른 Python/Jupyter 커널이 VRAM을 점유하지 않는지
확인합니다. 4GB 환경에서는 두 어댑터를 동시에 올리지 않고 한 번에 하나씩 학습합니다.

## 4GB용 핵심 설정

| 항목 | 설정 |
|---|---|
| 양자화 | NF4 4-bit + double quantization |
| compute dtype | bfloat16 |
| LoRA | rank 8, alpha 16 |
| 대상 모듈 | `q_proj`, `v_proj` |
| micro batch | 1 |
| gradient accumulation | 4 |
| 최대 길이 | 192 tokens |
| optimizer | paged AdamW 8-bit |
| 메모리 절약 | gradient checkpointing, cache 비활성화 |

## 제출 시 비교할 내용

각 task의 `data/*_test.jsonl`은 학습에 사용하지 않습니다. 동일한 입력에 대해 다음 세
조건을 비교합니다.

1. 4-bit base 모델
2. base 모델 + 상세 system prompt
3. 학습한 LoRA adapter (별도 system prompt 없음)

답장 작성기는 `2~3문장`, `정보 보존`, `명확한 의사 표현`, `부가 설명 없음`을 보고,
냉장고 도우미는 `지정 형식`, `입력 재료 활용`, `추가 핵심 재료 2개 이하`,
`3~4단계 조리법`을 확인합니다.
