import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

load_dotenv()

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_texts(chunks, embeddings)

def answer_question(knowledge_base, question):
    docs = knowledge_base.similarity_search(question)
    llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name='llama3-70b-8192'
    )
    chain = load_qa_chain(llm, chain_type='stuff')
    return chain.run(input_documents=docs, question=question)
