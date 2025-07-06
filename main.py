import streamlit as st
import pandas as pd
from io import StringIO
from backend.file_processing import extract_text_from_pdf, extract_text_from_txt
from backend.qa_pipeline import (
    process_text, build_vectorstore, answer_question, summarize_text, get_sentiment
)

from backend.insight_extraction import (
    get_top_keywords, get_named_entities,
    detect_headings, sentence_sentiments,
    detect_table_like_sections, generate_wordcloud
)

st.set_page_config(page_title='Conversational Data Assistant', layout='wide')
st.title("🧠 Conversational Data Assistant")

# -------------------- Session State Init -------------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "text_data" not in st.session_state:
    st.session_state.text_data = ""

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

# -------------------- File Upload -------------------- #
uploaded_file = st.file_uploader("📂 Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif file_type == "text/plain":
        text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    st.session_state.text_data = text

    st.success("✅ File successfully loaded!")

    # -------------------- Insights -------------------- #
    with st.expander("📊 Document Insights"):
        word_count = len(text.split())
        reading_time = round(word_count / 200)
        st.write(f"**Word Count:** {word_count}")
        st.write(f"**Estimated Reading Time:** {reading_time} min")

        # Sentiment
        sentiment, polarity = get_sentiment(text)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment)
        with col2:
            st.metric("Polarity Score", round(polarity, 3))

        # Key elements
        top_keywords = get_top_keywords(text)
        entities = get_named_entities(text)
        headings = detect_headings(text)
        sentiments = sentence_sentiments(text)
        tables = detect_table_like_sections(text)

        with st.expander("🔑 Top Keywords"):
            for word, count in top_keywords:
                st.markdown(f"- **{word}**: {count} times")

        with st.expander("🧠 Named Entities"):
            for ent_type, ent_vals in entities.items():
                st.markdown(f"**{ent_type}**")
                for name, count in ent_vals:
                    st.markdown(f"- {name} ({count})")

        if headings:
            with st.expander("📌 Headings"):
                for h in headings:
                    st.markdown(f"- {h}")

        with st.expander("🎭 Sentence-wise Sentiment (Top 5)"):
            for i, (sentence, score) in enumerate(sentiments[:5]):
                st.markdown(f"{i+1}. _{sentence}_ → **Polarity:** {score}")

        if tables:
            with st.expander("📋 Table-like Sections"):
                for block in tables:
                    st.code(block, language="text")

        if st.button("🖼️ Generate Word Cloud"):
            fig = generate_wordcloud(text)
            st.pyplot(fig)

    # -------------------- Summary -------------------- #
    if st.button("🪄 Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(text)
            st.subheader("📑 Summary")
            st.write(summary)

    # -------------------- VectorStore -------------------- #
    chunks = process_text(text)
    st.session_state.knowledge_base = build_vectorstore(chunks)

# -------------------- Q&A Chat Interface -------------------- #
if st.session_state.knowledge_base:
    st.markdown("---")
    st.subheader("💬 Ask Questions from the Document")

    user_question = st.text_input("Type your question here:")

    if st.button("Ask"):
        if user_question:
            with st.spinner("Thinking..."):
                answer = answer_question(st.session_state.knowledge_base, user_question)
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("AI", answer))

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 AI:** {message}")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
