"""Day 03 미니 프로젝트: Local LLM Serving Lab."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import streamlit as st

from benchmark import DEFAULT_PROMPTS, benchmark_levels
from capacity import (
    GIB,
    MIB,
    ModelArchitecture,
    available_kv_bytes,
    estimate_max_users,
    kv_bytes_per_token,
    kv_cache_bytes,
)


SERVER_PRESETS = {
    "Ollama": ("http://localhost:11434/v1", "qwen3:4b"),
    "vLLM": ("http://localhost:8000/v1", "Qwen/Qwen3-0.6B"),
    "직접 입력": ("http://localhost:8000/v1", ""),
}

st.set_page_config(page_title="Local LLM Serving Lab", page_icon="🧪", layout="wide")
st.title("🧪 Local LLM Serving Lab")
st.caption("동시 요청 성능을 측정하고 KV cache 기준 수용 인원을 계산합니다.")

benchmark_tab, capacity_tab, guide_tab = st.tabs(
    ["부하테스트", "KV cache 계산기", "결과 해석 가이드"]
)

with benchmark_tab:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("서버 및 실험 설정")
        server_name = st.selectbox("서버", list(SERVER_PRESETS))
        default_url, default_model = SERVER_PRESETS[server_name]
        base_url = st.text_input("OpenAI 호환 API 주소", value=default_url)
        model = st.text_input("모델", value=default_model)
        api_key = st.text_input("API key", value="not-needed", type="password")
        concurrency_levels = st.multiselect(
            "동시성 조건", [1, 2, 4, 8, 12, 16], default=[1, 2, 4, 8]
        )
        total_requests = st.slider("조건별 총 요청 수", 4, 40, 12, 4)
        max_tokens = st.slider("요청당 최대 생성 토큰", 16, 256, 128, 16)
        temperature = st.slider("temperature", 0.0, 1.5, 0.3, 0.1)
        timeout_s = st.slider("요청 timeout(초)", 10, 300, 120, 10)
        run_button = st.button(
            "부하테스트 실행",
            type="primary",
            width="stretch",
            disabled=not concurrency_levels or not model.strip(),
        )

    with right:
        st.info(
            "이 앱은 추론 서버를 실행하지 않습니다. 먼저 별도 터미널에서 "
            "`ollama serve` 또는 `vllm serve <모델>`을 실행하세요. 워밍업 "
            "요청 1회 후 각 동시성을 측정합니다."
        )
        if run_button:
            try:
                with st.spinner("서버에 부하를 주고 있습니다…"):
                    results = asyncio.run(
                        benchmark_levels(
                            base_url,
                            api_key,
                            model,
                            DEFAULT_PROMPTS,
                            sorted(concurrency_levels),
                            total_requests=total_requests,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            timeout_s=timeout_s,
                        )
                    )
                st.session_state["day03_results"] = [
                    result.to_dict() for result in results
                ]
                st.session_state["day03_settings"] = {
                    "server": server_name,
                    "base_url": base_url,
                    "model": model,
                    "total_requests": total_requests,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            except Exception as exc:
                st.error(f"측정을 시작하지 못했습니다: {exc}")

        if rows := st.session_state.get("day03_results"):
            table = pd.DataFrame(
                {
                    "동시성": [row["concurrency"] for row in rows],
                    "성공": [row["successful_requests"] for row in rows],
                    "실패": [row["failed_requests"] for row in rows],
                    "wall time(s)": [round(row["wall_time_s"], 2) for row in rows],
                    "p50(s)": [round(row["p50_latency_s"], 2) for row in rows],
                    "p95(s)": [round(row["p95_latency_s"], 2) for row in rows],
                    "req/s": [round(row["requests_per_s"], 2) for row in rows],
                    "tok/s": [round(row["tokens_per_s"], 1) for row in rows],
                }
            )
            st.subheader("측정 결과")
            st.dataframe(table, hide_index=True, width="stretch")

            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.markdown("**처리량**")
                st.line_chart(table.set_index("동시성")[["req/s", "tok/s"]])
            with chart_right:
                st.markdown("**지연시간**")
                st.line_chart(table.set_index("동시성")[["p50(s)", "p95(s)"]])

            errors = {
                row["concurrency"]: row["errors"] for row in rows if row["errors"]
            }
            if errors:
                with st.expander("오류 상세"):
                    st.json(errors)

            export = {
                "settings": st.session_state.get("day03_settings", {}),
                "results": rows,
            }
            st.download_button(
                "JSON 결과 다운로드",
                json.dumps(export, ensure_ascii=False, indent=2),
                file_name="llm_load_test.json",
                mime="application/json",
            )

with capacity_tab:
    st.subheader("모델 구조와 GPU 조건")
    st.caption("기본값은 Qwen3-0.6B의 구조를 기준으로 합니다.")
    arch_col, workload_col, gpu_col = st.columns(3)
    with arch_col:
        layers = st.number_input("Layer 수", 1, 200, 28)
        attention_heads = st.number_input("Attention head 수", 1, 256, 16)
        kv_heads = st.number_input("KV head 수", 1, 256, 8)
        head_dim = st.number_input("Head dimension", 1, 512, 128)
        dtype = st.selectbox("KV cache dtype", ["float16", "bfloat16", "fp8", "int8"])
    with workload_col:
        context_length = st.number_input(
            "사용자당 컨텍스트 토큰", 128, 1_048_576, 4096, step=128
        )
        concurrent_users = st.number_input("동시 사용자", 1, 10000, 10)
    with gpu_col:
        gpu_memory_gib = st.number_input("GPU 메모리(GiB)", 1.0, 640.0, 24.0)
        model_memory_gib = st.number_input("모델 점유 메모리(GiB)", 0.0, 640.0, 2.0)
        memory_utilization = st.slider("GPU 메모리 사용 한도", 0.1, 1.0, 0.9, 0.05)

    try:
        architecture = ModelArchitecture(
            int(layers), int(attention_heads), int(kv_heads), int(head_dim)
        )
        per_token = kv_bytes_per_token(architecture, dtype=dtype)
        per_user = kv_cache_bytes(
            architecture, int(context_length), dtype=dtype
        )
        total = kv_cache_bytes(
            architecture,
            int(context_length),
            int(concurrent_users),
            dtype=dtype,
        )
        available = available_kv_bytes(
            gpu_memory_gib,
            model_memory_gib,
            memory_utilization=memory_utilization,
        )
        max_users = estimate_max_users(
            architecture,
            int(context_length),
            gpu_memory_gib,
            model_memory_gib,
            dtype=dtype,
            memory_utilization=memory_utilization,
        )
        mha_per_user = kv_cache_bytes(
            architecture, int(context_length), dtype=dtype, use_mha=True
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("토큰당 KV", f"{per_token / 1024:.1f} KiB")
        metric_cols[1].metric("사용자당 KV", f"{per_user / MIB:.1f} MiB")
        metric_cols[2].metric("선택 인원의 KV", f"{total / GIB:.2f} GiB")
        metric_cols[3].metric("이론상 최대 사용자", f"{max_users:,}명")

        comparison = pd.DataFrame(
            {
                "Attention 방식": ["현재 모델(GQA/MQA)", "MHA 가정"],
                "KV head": [int(kv_heads), int(attention_heads)],
                "사용자당 KV(MiB)": [per_user / MIB, mha_per_user / MIB],
            }
        )
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption(
            f"KV cache 가용량은 약 {available / GIB:.2f} GiB입니다. 이 값은 모델 "
            "가중치와 KV cache만 단순화해 계산하므로 실제 서버 오버헤드는 별도입니다."
        )
    except ValueError as exc:
        st.error(str(exc))

with guide_tab:
    st.markdown(
        """
### 무엇을 보면 되나요?

- **p50**은 전형적인 사용자 경험, **p95**는 느린 상위 5% 요청의 경험입니다.
- 동시성을 올렸는데 `tok/s`가 더 늘지 않으면 서버가 포화된 것입니다.
- 처리량은 높아도 p95가 서비스 목표를 넘으면 허용 동시성을 낮춰야 합니다.
- 실패가 생기는 최초 동시성보다 충분히 낮은 값을 운영 한도로 잡아야 합니다.

### 비교 실험의 조건

vLLM FP16과 Ollama Q4 결과에는 엔진 차이뿐 아니라 정밀도 차이도 섞여 있습니다.
가능하면 같은 모델, 같은 양자화, 같은 프롬프트와 생성 토큰 수를 사용하세요.
Qwen3의 thinking 출력은 생성 토큰을 많이 사용하므로 두 서버의 thinking 설정도
맞추는 것이 좋습니다.
"""
    )
