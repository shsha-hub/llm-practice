import random
import streamlit as st
from openai import OpenAI

# ────────────────────────────────
# 0. 페이지 기본 설정
# ────────────────────────────────
st.set_page_config(
    page_title="AI 스무고개",
    page_icon="🎯",
    layout="centered",
)

# ────────────────────────────────
# 1. Pretendard 폰트 주입
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
MAX_QUESTIONS = 20

WORD_BANK = {
    "동물": ["코끼리", "펭귄", "고양이", "기린", "돌고래", "부엉이"],
    "음식": ["피자", "김밥", "초콜릿", "라면", "수박", "떡볶이"],
    "사물": ["우산", "핸드폰", "책상", "안경", "자전거", "선풍기"],
}

AVATARS = {"user": "🙋", "assistant": "🤖"}


def build_system_prompt(secret: str) -> str:
    return f"""
        너는 스무고개 게임의 출제자야. 지금 네가 마음속으로 정한 정답은 "{secret}" 야.
        이 정답은 어떤 경우에도 사용자에게 먼저 알려주면 안 돼.

        질문에는 두 종류가 있고, 종류에 맞게 답한다:
        1. 예/아니오로 답할 수 있는 질문 (예: "동물이야?") → "예", "아니오", "그런 편이야", "잘 모르겠어" 중 하나로 짧게 답한다.
        2. 색깔·크기·용도·사는 곳처럼 열린 정보를 묻는 질문 (예: "무슨 색이야?", "어디서 볼 수 있어?") → 정답 단어 자체는 말하지 않으면서, 사실에 맞는 정보를 한두 단어~한 문장으로 짧게 알려준다.

        그 외 규칙:
        3. 사용자가 정답을 직접 맞히면(정답 단어를 말하면) "정답이야!"라고 축하하고 정답이 "{secret}"였다는 걸 알려준다.
        4. 사용자가 "포기"라고 하거나 정답을 못 맞히고 질문 기회가 다 떨어지면, 정답이 "{secret}"였다는 걸 알려주고 짧게 힌트가 될 만한 설명을 덧붙인다.
        5. 어떤 경우에도 정답 단어 자체를 답변 문장 속에 그대로 흘리지 않는다 (2번 규칙으로 정보를 줄 때도 마찬가지).
        6. 대답은 한국어로, 한두 문장 이내로 짧게 한다.
    """


# ────────────────────────────────
# 2. 새 게임 시작 함수
# ────────────────────────────────
def start_new_game(category: str):
    pool = WORD_BANK[category] if category != "전체" else sum(WORD_BANK.values(), [])
    secret = random.choice(pool)
    st.session_state.secret = secret
    st.session_state.messages = [{"role": "system", "content": build_system_prompt(secret)}]
    st.session_state.game_active = True


if "game_active" not in st.session_state:
    st.session_state.game_active = False

# ────────────────────────────────
# 3. 사이드바 — 카테고리 선택 · 진행 정보
# ────────────────────────────────
with st.sidebar:
    st.header("🎮 게임 설정")
    category = st.selectbox("카테고리", ["전체", "동물", "음식", "사물"])

    if st.button("🔄 새 게임 시작", use_container_width=True):
        start_new_game(category)
        st.rerun()

    st.divider()

    if st.session_state.game_active:
        question_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
        remaining = max(MAX_QUESTIONS - question_count, 0)
        st.metric("남은 질문 수", f"{remaining} / {MAX_QUESTIONS}")
        st.progress(question_count / MAX_QUESTIONS)

        st.divider()
        if st.button("🏳️ 포기하고 정답 보기", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "포기할게, 정답 알려줘."})
            st.session_state.force_answer = True
            st.rerun()

# ────────────────────────────────
# 4. 헤더
# ────────────────────────────────
st.title("🎯 AI 스무고개")
st.markdown('<p class="subtitle">AI가 마음속으로 정한 답을 예/아니오 질문으로 맞혀 보세요 (최대 20문제)</p>', unsafe_allow_html=True)

# ────────────────────────────────
# 5. 게임 시작 전 안내 화면
# ────────────────────────────────
if not st.session_state.game_active:
    st.info("왼쪽 사이드바에서 카테고리를 고르고 **새 게임 시작**을 눌러 주세요.")
    st.stop()

# ────────────────────────────────
# 6. 대화 내역 렌더링 (system 메시지는 숨김 — 정답 보호)
# ────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]).write(msg["content"])

question_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
game_over = question_count >= MAX_QUESTIONS

# ────────────────────────────────
# 7. 강제 정답 공개(포기 버튼) 응답 처리
# ────────────────────────────────
if st.session_state.get("force_answer"):
    st.session_state.force_answer = False
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("정답을 공개하는 중..."):
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=st.session_state.messages,
                max_completion_tokens=200,
            )
            answer = response.choices[0].message.content
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.game_active = False

# ────────────────────────────────
# 8. 일반 질문 입력 처리
# ────────────────────────────────
if game_over:
    st.warning("질문 기회를 모두 사용했어요. 사이드바에서 정답을 확인해 보세요!")
else:
    prompt = st.chat_input("질문을 입력하세요 (예/아니오 질문이든, '무슨 색이야?' 같은 질문이든 OK)")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar=AVATARS["user"]).write(prompt)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("생각하는 중..."):
                response = client.chat.completions.create(
                    model=API_MODEL,
                    messages=st.session_state.messages,
                    max_completion_tokens=200,
                )
                answer = response.choices[0].message.content
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})