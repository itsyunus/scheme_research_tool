# PDF Scheme Research Assistant

A Streamlit prototype for exploring uploaded PDF documents with retrieval-augmented question answering. It extracts text, splits it into chunks, builds a local FAISS index, and uses OpenAI embeddings and a chat model to answer questions.

## How it works

1. Upload a PDF in the Streamlit sidebar.
2. Choose **Process Document** to create the vector index.
3. Ask a question about the uploaded document.

The current interface shows an answer and a simple extracted summary. It does not provide page citations or validate answers against a benchmark, so check important details against the source PDF.

## Run locally

Requirements: Python 3.9 or later and an OpenAI API key.

```bash
git clone https://github.com/itsyunus/scheme_research_tool.git
cd scheme_research_tool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
streamlit run main.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` and set the key with `$env:OPENAI_API_KEY = "your-api-key"`.

The application writes its FAISS data to `faiss_store_openai.pkl` in the working directory. Keep API keys and document data out of commits. Uploaded PDF text is sent to OpenAI for embeddings and answer generation; do not upload documents unless you are authorized to share them with that service. Do not load pickle files from untrusted sources.

## Current limitations

- Answers are generated from retrieved text and may be incomplete or incorrect.
- The interface does not show source page references.
- The index is local and is rebuilt when a document is processed.
