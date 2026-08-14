import streamlit as st
from openai import OpenAI

# ────────────────────────────────
# 0. 페이지 기본 설정 — 탭 제목·아이콘·레이아웃
# ────────────────────────────────
st.set_page_config(
    page_title="AI 영어 상황극 튜터",
    page_icon="🗣️",
    layout="centered",
)

# ────────────────────────────────
# 1. Pretendard 폰트 + 커스텀 CSS
# ────────────────────────────────
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Pretendard', sans-serif !important;
}
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}
.block-container { padding-top: 2rem; }
[data-testid="stChatMessage"] {
    padding: 0.9rem 1.1rem;
    border-radius: 14px;
    margin-bottom: 0.4rem;
}
.subtitle {
    color: var(--text-color, #6b6b6b);
    font-size: 0.95rem;
    margin-top: -0.6rem;
    margin-bottom: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

client = OpenAI()
API_MODEL = "gpt-5.4-nano"

SYSTEM_PROMPT = """
너는 친절한 원어민 영어 회화 튜터야.
사용자가 '공항', '식당', '호텔' 등 원하는 대화 상황을 입력하면, 네가 먼저 해당 상황에 맞는 원어민 직원의 역할로 영어 대화를 시작해.
사용자가 영어로 답변하면 문법이나 표현이 자연스러운지 짧게 한국어로 교정(Feedback)을 해주고, 대화를 계속 이어나가.
만약 사용자가 한국어로 질문하거나 모르는 단어를 물어보면 친절하게 한국어로 알려줘.
"""

AVATARS = {"user": "🧑‍💻", "assistant": "🗽"}

# ────────────────────────────────
# 2. 세션 상태 초기화
# ────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# ────────────────────────────────
# 3. 사이드바 — 상황 프리셋 · 초기화 · 진행 정보
# ────────────────────────────────
with st.sidebar:
    st.header("🎬 상황 고르기")
    st.caption("눌러서 바로 시작해 보세요")

    presets = ["✈️ 공항 입국심사", "🍔 패스트푸드점", "🏨 호텔 체크인", "🛍️ 옷가게에서 쇼핑"]
    preset_clicked = None
    for p in presets:
        if st.button(p, use_container_width=True):
            preset_clicked = p.split(" ", 1)[1]  # 이모지 떼고 텍스트만

    st.divider()

    turn_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.metric("진행한 턴", f"{turn_count}턴")

    st.divider()
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

# ────────────────────────────────
# 4. 헤더
# ────────────────────────────────
st.title("🗣️ AI 영어 상황극 튜터")
st.markdown('<p class="subtitle">상황을 입력하면 원어민 역할극이 시작돼요 · 표현은 자동으로 교정해 드려요</p>', unsafe_allow_html=True)

# ────────────────────────────────
# 5. 대화 내역 렌더링
# ────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]).write(msg["content"])

# ────────────────────────────────
# 6. 입력 처리 — 사이드바 프리셋 클릭 or 직접 입력
# ────────────────────────────────
prompt = st.chat_input("원하는 상황을 입력해 보세요. (예: 패스트푸드점, 뉴욕 공항 입국심사)")
if preset_clicked:
    prompt = preset_clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar=AVATARS["user"]).write(prompt)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("튜터가 답변을 준비하고 있어요..."):
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=st.session_state.messages,
                max_completion_tokens=400,
            )
            answer = response.choices[0].message.content
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
