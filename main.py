import streamlit as st
from backend.file_processing import extract_text_from_pdf, extract_text_from_txt
from backend.qa_pipeline import process_text, build_vectorstore, answer_question

def main():
    st.set_page_config(page_title='Ask your PDF/TXT')
    st.header('Ask your PDF or TXT')

    uploaded_file = st.file_uploader('Upload your PDF or TXT file', type=['pdf', 'txt'])

    if uploaded_file is not None:
        if uploaded_file.type == 'application/pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == 'text/plain':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error('Unsupported file format')
            return

        chunks = process_text(text)
        knowledge_base = build_vectorstore(chunks)

        user_question = st.text_input('Ask a question about your document:')
        if user_question:
            response = answer_question(knowledge_base, user_question)
            st.write(response)

if __name__ == '__main__':
    main()
