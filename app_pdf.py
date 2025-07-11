## importing packages

import streamlit as st           # For building the web app
import pandas as pd              # For handling tabular data
import base64                    # For enabling file downloads
import os                        # For handling environment vars
import requests                  # For any API calls

from PyPDF2 import PdfReader     # To read PDF files
from datetime import datetime    # To get timestamps

# LangChain-related imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Function: Extract text from PDFs
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function: Split text into chunks
def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks

# Function: Generate embeddings and store in ChromaDB
def get_embeddings(chunks, model_name, api_key):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    else:
        raise ValueError("Model not supported")

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vector_store.persist()

    return vector_store

# Function: Create QA chain
def create_qa_chain(model_name, vector_store=None, api_key=None):
    if model_name == "Google AI":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
    else:
        raise ValueError("Model not supported")

    prompt = PromptTemplate(
        template="""Answer the question based on the context below. \
If you can't find the answer, say \"I don't have enough information.\"

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    return retrieval_qa

# Function: User Q&A interaction
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please provide the API key and upload a PDF document to proceed.")
        return

    with st.spinner("Processing PDFs and generating answer..."):
        text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(text, model_name)
        vector_store = get_embeddings(text_chunks, model_name, api_key)
        chain = create_qa_chain(model_name, vector_store, api_key)
        response = chain({"query": user_question})

        user_question_output = user_question
        response_output = response['result']

        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((
            user_question_output,
            response_output,
            model_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ", ".join(pdf_names)
        ))

    display_chat_interface(user_question_output, response_output, conversation_history)

# Function: Chat display
def display_chat_interface(user_question_output, response_output, conversation_history):
    st.markdown(
        f"""
        <style>
            .chat-message {{ padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; }}
            .chat-message.user {{ background-color: #2b313e; }}
            .chat-message.bot {{ background-color: #475063; }}
            .chat-message .avatar {{ width: 20%; }}
            .chat-message .avatar img {{ max-width: 78px; max-height: 78px; border-radius: 50%; object-fit: cover; }}
            .chat-message .message {{ width: 80%; padding: 0 1.5rem; color: #fff; }}
            .chat-message .info {{ font-size: 0.8rem; margin-top: 0.5rem; color: #ccc; }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
            </div>
            <div class="message">{response_output}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(conversation_history) > 1:
        for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history[:-1]):
            st.markdown(
                f"""
                <div class="chat-message user">
                    <div class="avatar">
                        <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                    </div>
                    <div class="message">{question}</div>
                </div>
                <div class="chat-message bot">
                    <div class="avatar">
                        <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                    </div>
                    <div class="message">{answer}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history,
                         columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side.")

# Main Streamlit UI
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.header("Chat with multiple PDFs 📚")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.sidebar.markdown("""
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/)
        [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)
    """)

    model_name = st.sidebar.radio("Select the Model:", ("Google AI",))

    api_key = None
    if model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
        st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        if not api_key:
            st.sidebar.warning("Please enter your Google API Key to proceed.")
            return

    with st.sidebar:
        st.title("Menu:")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset")
        clear_button = col1.button("Clear Last")

        if reset_button:
            st.session_state.conversation_history = []
            st.rerun()

        if clear_button and st.session_state.conversation_history:
            st.session_state.conversation_history.pop()
            st.rerun()

        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=['pdf']
        )

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    st.success("PDFs uploaded successfully! You can now ask questions.")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and pdf_docs and api_key:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
    elif user_question and not pdf_docs:
        st.warning("Please upload PDF files first.")
    elif user_question and not api_key:
        st.warning("Please enter your Google API Key first.")

if __name__ == "__main__":
    main()


