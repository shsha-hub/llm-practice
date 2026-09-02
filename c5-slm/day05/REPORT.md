# QLoRA 기반 멀티 생활 도우미 실험 보고서

## 1. 실험 목적

4GB VRAM 환경에서 2B 언어모델을 4비트로 양자화하고, 하나의 base 모델에 목적이 다른
LoRA 어댑터 두 개를 학습했다.

1. 곤란한 상황의 답장 작성기
2. 냉장고 재료 소진 도우미

이 실험의 목표는 새로운 지식을 많이 주입하는 것이 아니라, 짧은 데이터로 원하는
응답 형식과 행동 규칙을 안정적으로 학습할 수 있는지 확인하는 것이다.

## 2. 문제 분석과 설계 변경

강의 원본 설정은 rank 16의 LoRA를 모든 linear layer에 적용하고 micro batch 2,
max length 512를 사용했다. RTX 3050 Laptop의 VRAM은 4GB이고 당시 다른 커널까지
메모리를 점유해 모델 초기화 단계에서 CPU offload 오류가 발생했다.

또한 Mi:dm 토크나이저의 기본 chat template는 약 500토큰의 system prompt를 자동으로
삽입했다. 원본 학습 샘플은 535~552토큰이므로 max length 512에서는 assistant 정답이
잘릴 수 있었다. 이를 해결하기 위해 다음과 같이 변경했다.

- 커널 재시작 후 학습 시작 시 여유 VRAM 3.2GiB 이상 확인
- NF4 4-bit double quantization
- LoRA rank 8, alpha 16
- `q_proj`, `v_proj`만 학습
- micro batch 1, gradient accumulation 4
- gradient checkpointing과 paged AdamW 8-bit 사용
- 긴 기본 template 대신 짧은 native prompt/completion 형식 사용
- assistant completion에만 loss 적용
- 최대 길이 192토큰

## 3. 데이터 구성

각 task는 학습 40개, 검증 8개, 미학습 테스트 5개로 구성했다. 테스트 입력은 학습과
검증 데이터에 포함하지 않았으며 `scripts/validate.py`로 split 간 입력 중복과 토큰 길이를
검사했다.

| Task | Train | Validation | Test | 최대 학습 토큰 |
|---|---:|---:|---:|---:|
| 답장 작성기 | 40 | 8 | 5 | 71 |
| 냉장고 도우미 | 40 | 8 | 5 | 113 |

## 4. 학습 결과

두 어댑터 모두 학습 가능한 파라미터는 3,342,336개로 전체 모델의 0.1448%였다.

| Task | 시간 | 최고 할당 VRAM | Train loss | Validation loss (epoch 1 → 3) |
|---|---:|---:|---:|---:|
| 답장 작성기 | 72.8초 | 2.82GiB | 2.1353 | 2.218 → 2.095 → 2.041 |
| 냉장고 도우미 | 72.7초 | 2.83GiB | 1.4201 | 1.531 → 1.161 → 1.105 |

두 task 모두 validation loss가 매 epoch 감소했고 4GB VRAM 안에서 OOM 없이
완료됐다.

## 5. 미학습 테스트 결과

동일한 테스트 입력에 다음 세 조건을 적용했다.

1. 4-bit base
2. base + 상세 system prompt
3. LoRA adapter + 별도 system prompt 없음

모든 조건에 동일한 최대 생성 길이와 task별 종료 규칙을 적용했다.

| Task | Base | Prompted base | Adapter |
|---|---:|---:|---:|
| 답장 형식·의도 준수 | 80% | 95% | **100%** |
| 냉장고 형식·재료 준수 | 48% | 52% | **96%** |

답장 adapter는 테스트 5개 모두에서 필요한 금액, 날짜 또는 요청 목적을 보존했다.
예를 들어 관리실에 경보음 점검을 요청하는 입력에서 base는 관리실이 주민에게 답하는
방향으로 화자를 혼동했지만, adapter는 주민이 보낼 수 있는 두 문장 요청문을 생성했다.

```text
새벽마다 주차장 경보음이 반복되어 잠을 설칩니다.
이번 주 안에 원인을 점검해 주시면 감사하겠습니다.
```

냉장고 adapter는 모든 테스트에서 `추천 → 추가 재료 → 조리 → 소진 재료` 구조를
생성했다. base와 prompted base는 기존 모델의 마크다운 레시피 형식을 사용하거나 생성
한도 안에 답을 끝내지 못했다.

## 6. 한계와 개선 방향

냉장고 테스트 한 건은 조리 단계가 2개여서 3~4단계 규칙을 지키지 못했다. 또 남은
불고기 테스트에서는 `추가 재료`에 달걀만 적고 조리 단계에서 밀가루를 사용했다. 이는
형식은 잘 학습했지만 40개 데이터만으로 재료 제약을 완전히 일반화하지 못했다는 뜻이다.

개선 방법은 다음과 같다.

- 조리하지 않는 입력 재료, 미기재 추가 재료 같은 hard negative 사례 보강
- 데이터에 `사용 가능 기본 양념` 목록 명시
- 생성 후 입력 재료와 추가 재료를 대조하는 규칙 기반 validator 결합
- 테스트셋을 20개 이상으로 늘려 음식 종류별 성능 측정

이번 결과는 LoRA가 작은 데이터에서도 출력 구조와 말투를 효율적으로 학습하지만,
사실·제약의 완전한 보장은 별도 검증 로직이 필요하다는 점을 보여준다.

## 7. 재현 파일

- `scripts/train.py`: 두 task 공용 4GB QLoRA 학습
- `scripts/compare.py`: base / prompted base / adapter 비교
- `scripts/evaluate.py`: 형식과 핵심 정보 준수율 계산
- `scripts/validate.py`: 데이터 수량·중복·토큰 길이 검사
- `life_assistant/config.py`: task 경로와 prompt/completion 구성
- `life_assistant/inference.py`: task별 생성 종료와 출력 정리
- `notebooks/01_train_qlora_workbook.ipynb`: 셀 단위 QLoRA 학습 복습
- `notebooks/02_compare_adapters_workbook.ipynb`: 세 조건 추론 비교 복습
- `results/training_metrics.json`: 실제 학습 시간, VRAM, loss 기록
- `outputs/comparison-*.json`: 미학습 테스트 원본 비교 결과

## 8. Streamlit 시연

`app.py`에서 base 모델을 4-bit로 한 번만 GPU에 올린 뒤 `reply`, `fridge` 어댑터를
동시에 등록한다. 사용자가 task를 선택하면 `set_adapter()`로 작은 어댑터만 전환하며,
동일한 입력의 base, prompted base, adapter 출력을 세 탭에서 비교한다. 각 결과에는
생성 시간과 task별 형식 준수 검사도 함께 표시한다.

```bash
.venv/bin/streamlit run day05/app.py
```

실제 검증에서 두 어댑터가 한 base에 정상 등록됐고, 답장 생성 시 GPU 할당 메모리는
약 1.47GiB였다. Streamlit UI 테스트와 로컬 health endpoint 검사도 통과했다.
