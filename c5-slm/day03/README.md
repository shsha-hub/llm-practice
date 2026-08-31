# Day 03 미니 프로젝트: Local LLM Serving Lab

OpenAI 호환 API를 제공하는 Ollama 또는 vLLM 서버에 비동기 부하를 주고,
동시성별 처리량과 tail latency를 비교하는 Streamlit 앱입니다. 모델 구조와
GPU 조건을 입력하면 KV cache 메모리와 이론상 최대 동시 사용자 수도 계산합니다.

## 실행

이 앱은 추론 서버를 직접 실행하지 않고 이미 실행 중인 OpenAI 호환 API에
요청을 보냅니다. 따라서 Ollama 또는 vLLM 서버를 별도 터미널에서 먼저
실행해야 합니다.

### Ollama로 테스트

```bash
ollama serve
ollama pull qwen3:4b
```

앱에서 서버를 `Ollama`, API 주소를 `http://localhost:11434/v1`로 선택합니다.

### vLLM으로 테스트

현재 가상환경에 vLLM이 없다면 먼저 설치합니다. vLLM은 하드웨어와 CUDA
환경의 영향을 받으므로 설치가 실패하면 공식 설치 문서에서 자신의 GPU에
맞는 방법을 확인하세요.

```bash
uv pip install --python .venv/bin/python vllm
```

별도 터미널에서 OpenAI 호환 서버를 실행합니다.

```bash
cd /home/student/llm-practice/c5-slm
.venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto
```

모델 다운로드와 로딩이 끝난 뒤 다음 명령에 모델 정보가 반환되면 준비된
것입니다.

```bash
curl http://localhost:8000/v1/models
```

앱에서 서버를 `vLLM`, API 주소를 `http://localhost:8000/v1`, 모델을
`Qwen/Qwen3-0.6B`로 선택합니다. API key는 서버를 `--api-key` 없이 실행한
경우 `not-needed` 그대로 두어도 됩니다.

### Streamlit 앱 실행

추론 서버를 켜 둔 상태에서 다른 터미널을 열어 앱을 실행합니다.

```bash
cd /home/student/llm-practice/c5-slm
.venv/bin/streamlit run day03/streamlit_app.py
```

## 테스트

서버가 없어도 집계 로직과 KV cache 계산을 테스트할 수 있습니다.

```bash
.venv/bin/python -m unittest discover -s day03 -p 'test_*.py' -v
```

## 측정 지표

- `p50`: 성공 요청 중 중앙 지연시간
- `p95`: 느린 상위 5% 경계의 지연시간
- `req/s`: 전체 측정 시간 동안 완료한 성공 요청 수
- `tok/s`: 전체 측정 시간 동안 생성한 completion token 수
- 성공/실패: timeout을 포함한 요청 결과

각 실험은 짧은 워밍업 요청 후 시작합니다. 동시성보다 조건별 총 요청 수가
작으면 최소한 동시성만큼 요청하도록 자동 보정합니다.

## KV cache 계산

토큰당 KV cache 크기는 다음 식으로 계산합니다.

```text
layers × kv_heads × head_dim × 2(K,V) × dtype_bytes
```

Qwen3-0.6B(`28 layers`, `8 KV heads`, `head_dim 128`)를 FP16으로 계산하면
토큰당 114,688 bytes이고, 사용자당 4,096토큰은 448 MiB입니다.

최대 사용자 수는 다음과 같이 단순화한 이론값입니다.

```text
(GPU 메모리 × 사용 한도 - 모델 점유 메모리) / 사용자당 KV cache
```

실제 서버에는 activation, CUDA graph, allocator fragmentation 등의 메모리가
추가로 필요하므로 운영 한도는 이 값보다 낮게 잡아야 합니다.

## 비교할 때 주의할 점

- 같은 모델, 양자화, 프롬프트, 생성 토큰 수를 맞춰야 엔진을 공정하게 비교할 수 있습니다.
- vLLM FP16과 Ollama Q4 비교에는 엔진 외에 정밀도 차이도 포함됩니다.
- 짧은 1회 실험은 편차가 큽니다. 보고서용 결과는 각 조건을 여러 번 실행하세요.
- non-streaming 요청이므로 현재 latency는 전체 응답 시간입니다. TTFT는 측정하지 않습니다.
