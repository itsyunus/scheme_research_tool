# PDF Scheme Document Semantic Search & Question Answering

A Retrieval-Augmented Generation (RAG) assistant for complex regulatory and government scheme documents. Extracts, indexes, and queries unstructured PDF publications using Streamlit, LangChain, FAISS, and OpenAI embeddings.

## Pipeline Architecture

1. Ingestion: Loads PDF publications using PyPDFLoader.
2. Chunking & Indexing: Splits text into overlapping semantic passages using RecursiveCharacterTextSplitter and indexes vector embeddings into a local FAISS database.
3. Query Execution: Retrieves top-k matching document passages and constructs prompt contexts for conversational LLM response generation.
4. User Interface: Streamlit web client supporting real-time document upload, processing status, and interactive query evaluation.

## Installation and Setup

```bash
pip install -r requirements.txt
streamlit run main.py
```
