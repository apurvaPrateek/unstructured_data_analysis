import streamlit as st
import pandas as pd
from io import StringIO
from backend.file_processing import extract_text_from_pdf, extract_text_from_txt, is_csv_content
from backend.qa_pipeline import process_text, build_vectorstore, answer_question, summarize_text, get_sentiment

def main():
    st.set_page_config(page_title='Ask your PDF/TXT', layout='centered')
    st.header('📄 Ask your PDF or TXT')

    uploaded_file = st.file_uploader('Upload your PDF or TXT file', type=['pdf', 'txt'])

    if uploaded_file is not None:
        # check structured or not?
        if uploaded_file.type == 'application/pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == 'text/plain':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error('Unsupported file format')
            return
            # st.stop()

        # if is_csv_content(text):
        #     df = pd.read_csv(StringIO(text))
        #     handle_structured_csv(df)
        # else:
        #     handle_unstructured_text(text)

        # ------------------------------------------------------
        # ------------------ Document Insights -----------------
        # ------------------------------------------------------

        # ------------------ TEXT STATISTICS -------------------
        st.subheader("📊 Document Insights")
        word_count = len(text.split())
        reading_time = round(word_count / 200)
        st.write(f"**Word Count:** {word_count}")
        st.write(f"**Estimated Reading Time:** {reading_time} min")

        # ------------------ SENTIMENT ANALYSIS -------------------
        st.subheader("🩺 Sentiment Analysis")
        sentiment, polarity = get_sentiment(text)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Sentiment", value=sentiment)
        with col2:
            st.metric(label="Polarity Score", value=round(polarity, 3))

        # -------------------- KEY FEATURES ---------------------
        with st.expander("📌 Key Elements Extracted"):
            from backend.insight_extraction import (
                get_top_keywords, get_named_entities,
                detect_headings, sentence_sentiments,
                detect_table_like_sections, generate_wordcloud
            )

            # After uploading and extracting text...
            top_keywords = get_top_keywords(text)
            with st.expander("🔑 Top Keywords"):
                for word, count in top_keywords:
                    st.markdown(f"- **{word}**: {count} times")

            entities = get_named_entities(text)
            with st.expander("🧠 Named Entities"):
                for entity_type, items in entities.items():
                    st.markdown(f"**{entity_type}**")
                    for name, count in items:
                        st.markdown(f"- {name} ({count} times)")
            
            headings = detect_headings(text)
            if headings:
                with st.expander("📌 Detected Headings"):
                    for h in headings:
                        st.markdown(f"- {h}")
            # else:
            #     st.info("No clear headings found.")

            sentiments = sentence_sentiments(text)
            with st.expander("🎭 Sentence-wise Sentiment"):
                for i, (sentence, score) in enumerate(sentiments[:5]):  # limit to 5 for readability
                    st.markdown(f"{i+1}. _{sentence}_ → **Polarity:** {score}")

            tables = detect_table_like_sections(text)
            if tables:
                with st.expander("📋 Table-like Sections (Top 3)"):
                    for i, block in enumerate(tables):
                        st.code(block, language='text')

            if st.button("🧠 Generate Word Cloud"):
                fig = generate_wordcloud(text)
                st.pyplot(fig)


        # ------------------ GENERATE SUMMARY -------------------
        if st.button("🪄 Generate Summary"):
            with st.spinner("Generating document summary ..."):
                summary = summarize_text(text)
            with st.container(border=True):
                st.subheader("📑 Summary")
                st.write(summary)
        

        # ------------------------------------------------------
        # ------------------------ Q&A -------------------------
        # ------------------------------------------------------

        # ------------------ Q&A SECTION -------------------
        st.subheader("❓ Ask a Question")
        chunks = process_text(text)
        knowledge_base = build_vectorstore(chunks)

        user_question = st.text_input('Ask a question about your document:')
        if user_question:
            with st.spinner("Fetching answer ..."):
                response = answer_question(knowledge_base, user_question)
            st.success(response)

if __name__ == '__main__':
    main()
