Certainly! Here's a **professional `README.md`** version without emojis:

---

# Ask Anything From PDFs

This is a **Streamlit web application** that allows users to **upload PDF files** and **ask questions** based on their content. The application uses **LangChain**, **Google Generative AI**, and **Chroma vector store** to extract, embed, and retrieve answers in an intelligent and contextual manner.

---

## Features

* Upload one or more PDF documents
* Ask natural language questions about the contents of the PDFs
* Automatically extracts and processes PDF text
* Uses Google Generative AI to generate embeddings
* Embeddings are stored in ChromaDB (a vector database)
* Answers questions based on vector similarity and LLM responses
* Displays conversation history with timestamps
* Provides an option to download the entire chat history as a CSV
* Simple and clean chat interface

---

## Tech Stack

| Purpose        | Library/Tool                      |
| -------------- | --------------------------------- |
| Web App        | `streamlit`                       |
| PDF Reading    | `PyPDF2`                          |
| Text Splitting | `langchain.text_splitter`         |
| Embeddings     | `GoogleGenerativeAIEmbeddings`    |
| Vector Store   | `Chroma`                          |
| Language Model | `ChatGoogleGenerativeAI` (Gemini) |
| Q\&A Chain     | `RetrievalQA`                     |
| Chat Interface | HTML/CSS in Streamlit             |

---

## Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

2. **Create a Virtual Environment**

```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the App**

```bash
streamlit run app.py
```

---

## Getting a Google API Key

1. Visit: [https://ai.google.dev/](https://ai.google.dev/)
2. Log in with your Google account
3. Create and enable the **Generative AI API**
4. Generate an API key and copy it
5. Enter the key into the app sidebar input field

---

## Usage Guide

1. Launch the app using the command above.
2. Enter your **Google API key** in the sidebar.
3. Upload one or more PDF documents.
4. Click **Submit & Process** to prepare the documents.
5. Enter a question in the main input field.
6. View the model's answer along with the previous Q\&A history.
7. Download the entire conversation from the sidebar if needed.

---

## Menu Options

* **Clear Last**: Removes the last question-answer pair from the history.
* **Reset**: Clears the entire session's conversation history.
* **Download**: Allows you to download the chat as a CSV file.

---

## Example Use Cases

* Extracting key insights from financial reports
* Summarizing legal documents
* Academic paper review
* Internal policy clarification

---

## File Structure

```
├── app.py                # Main Streamlit application
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
├── chroma_db/            # Persistent vector database (auto-generated)
```

---

## System Overview

1. **PDF Parsing**: Extracts all text from uploaded PDFs using PyPDF2.
2. **Chunking**: Breaks down large text into manageable pieces for vectorization.
3. **Embedding**: Generates vector embeddings using Google's Embedding Model.
4. **Vector Store**: Saves vectors in ChromaDB for similarity search.
5. **Q\&A Generation**: Uses a prompt-based RetrievalQA chain to fetch context and generate answers using Gemini LLM.

---
