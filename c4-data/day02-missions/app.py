"""Streamlit 기반 LLM 개념 학습 챗봇."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from study_core import (
    CHROMA_DIR,
    PROJECT_DIR,
    TOPICS,
    Grade,
    Quiz,
    answer_question,
    create_quiz,
    get_llm,
    get_store,
    grade_answer,
    retrieve,
)


load_dotenv(PROJECT_DIR / ".env")
st.set_page_config(page_title="LLM Study Mate", page_icon="📘", layout="wide")


@st.cache_resource
def cached_store():
    return get_store()


@st.cache_resource
def cached_llm():
    return get_llm()


def init_state() -> None:
    defaults = {
        "messages": [],
        "topic_answer": None,
        "topic_docs": [],
        "topic_answer_topic": None,
        "quiz": None,
        "quiz_docs": [],
        "quiz_topic": None,
        "quiz_difficulty": None,
        "grade": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_sources(docs) -> None:
    if not docs:
        return
    with st.expander("참고한 학습 자료"):
        for number, doc in enumerate(docs, 1):
            heading = doc.metadata.get("h3") or doc.metadata.get("h2") or doc.metadata.get("h1")
            st.markdown(f"**[{number}] {heading or '학습 자료'}**")
            st.caption(f"{doc.metadata.get('source', '')} · {doc.metadata.get('chunk_id', '')}")
            st.text(doc.page_content)


def reset_learning() -> None:
    for key in (
        "messages", "topic_answer", "topic_docs", "topic_answer_topic", "quiz",
        "quiz_docs", "quiz_topic", "quiz_difficulty", "grade",
    ):
        del st.session_state[key]
    st.rerun()


init_state()
st.title("📘 LLM Study Mate")
st.caption("학습 자료를 검색해 설명하고, 이해 확인 문제와 피드백을 제공하는 근거 기반 튜터")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 없습니다. `.env.example`을 참고해 `.env` 파일을 설정하세요.")
    st.stop()

store = cached_store()
if not CHROMA_DIR.exists() or store._collection.count() == 0:
    st.error("학습 자료 색인이 없습니다. 먼저 `python ingest.py`를 실행하세요.")
    st.stop()

llm = cached_llm()

with st.sidebar:
    st.header("학습 설정")
    level = st.radio("설명 수준", ["입문", "일반", "심화"], horizontal=True)
    topic = st.selectbox("학습 주제", TOPICS)
    k = st.slider("검색할 자료 조각", 2, 6, 3)
    difficulty = st.select_slider("퀴즈 난이도", ["쉬움", "보통", "어려움"], value="보통")
    st.metric("색인된 조각", f"{store._collection.count():,}개")
    if st.button("새 학습 시작", use_container_width=True):
        reset_learning()
    st.caption("답변과 자동 채점은 틀릴 수 있으므로 아래 원문 근거를 함께 확인하세요.")

question_tab, topic_tab, quiz_tab = st.tabs(["💬 자유 질문", "📖 주제 학습", "✏️ 퀴즈"])

with question_tab:
    st.subheader("궁금한 개념을 질문하세요")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("docs"):
                show_sources(message["docs"])

    question = st.chat_input("예: RAG는 환각을 어떻게 줄이나요?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("학습 자료에서 근거를 찾고 있어요..."):
                docs = retrieve(store, f"현재 주제: {topic}\n질문: {question}", k=k)
                try:
                    answer = answer_question(llm, docs, question, level)
                except Exception as error:
                    st.error(f"답변 생성에 실패했습니다: {error}")
                else:
                    st.markdown(answer)
                    show_sources(docs)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "docs": docs}
                    )

with topic_tab:
    st.subheader(f"오늘의 주제: {topic}")
    if st.button("이 주제 설명하기", type="primary", use_container_width=True):
        with st.spinner("수준에 맞는 설명을 준비하고 있어요..."):
            docs = retrieve(store, f"{topic}의 정의, 원리, 예시와 한계", k=k)
            try:
                answer = answer_question(llm, docs, f"{topic}을 설명해 주세요.", level)
            except Exception as error:
                st.error(f"설명 생성에 실패했습니다: {error}")
            else:
                st.session_state.topic_answer = answer
                st.session_state.topic_docs = docs
                st.session_state.topic_answer_topic = topic

    if st.session_state.topic_answer:
        st.caption(f"설명을 만든 주제: {st.session_state.topic_answer_topic}")
        st.markdown(st.session_state.topic_answer)
        show_sources(st.session_state.topic_docs)
        st.info("설명을 자신의 말로 다시 말해본 뒤 퀴즈 탭에서 확인해 보세요.")

with quiz_tab:
    st.subheader(f"{topic} 이해 확인")
    if st.button("새 문제 만들기", type="primary", use_container_width=True):
        with st.spinner("학습 자료로 문제를 만들고 있어요..."):
            docs = retrieve(store, f"{topic}의 핵심 원리와 이해 확인 문제", k=k)
            try:
                quiz = create_quiz(llm, docs, topic, difficulty)
            except Exception as error:
                st.error(f"문제 생성에 실패했습니다: {error}")
            else:
                st.session_state.quiz = quiz.model_dump()
                st.session_state.quiz_docs = docs
                st.session_state.quiz_topic = topic
                st.session_state.quiz_difficulty = difficulty
                st.session_state.grade = None

    if st.session_state.quiz:
        quiz = Quiz.model_validate(st.session_state.quiz)
        st.caption(
            f"주제: {st.session_state.quiz_topic} · 난이도: {st.session_state.quiz_difficulty}"
        )
        st.markdown(f"### 문제\n{quiz.question}")
        with st.form("answer_form"):
            user_answer = st.text_area("내 답안", placeholder="자신의 말로 설명해 보세요.", height=140)
            submitted = st.form_submit_button("답안 제출", use_container_width=True)
        if submitted:
            if not user_answer.strip():
                st.warning("답안을 입력한 후 제출해 주세요.")
            else:
                with st.spinner("근거와 핵심 의미를 비교하고 있어요..."):
                    try:
                        grade = grade_answer(llm, st.session_state.quiz_docs, quiz, user_answer)
                    except Exception as error:
                        st.error(f"채점에 실패했습니다: {error}")
                    else:
                        st.session_state.grade = grade.model_dump()

        if st.session_state.grade:
            grade = Grade.model_validate(st.session_state.grade)
            st.metric("점수", f"{grade.score}점", grade.verdict)
            left, right = st.columns(2)
            with left:
                st.success(f"**잘한 점**\n\n{grade.good_point}")
            with right:
                st.warning(f"**빠진 핵심**\n\n{grade.missing_point}")
            st.markdown(f"**보완 설명**\n\n{grade.feedback}")
            with st.expander("모범 답안과 해설"):
                st.markdown(f"**모범 답안**\n\n{quiz.answer}")
                st.markdown(f"**해설**\n\n{quiz.explanation}")
                st.caption(f"핵심어: {', '.join(quiz.keywords)}")
            show_sources(st.session_state.quiz_docs)
