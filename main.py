import streamlit as st
from utils.processor import process_file, answer_query
import pickle

st.title("Scheme Research Assistant (PDF-based)")

uploaded_file = st.sidebar.file_uploader("Upload Scheme PDF", type=["pdf"])
process_btn = st.sidebar.button("Process Document")

if process_btn and uploaded_file:
    with st.spinner("Processing..."):
        faiss_index, summaries = process_file(uploaded_file)
        with open("faiss_store_openai.pkl", "wb") as f:
            pickle.dump((faiss_index, summaries), f)
        st.success("Processing complete!")

query = st.text_input("Ask a question about the scheme:")
if query:
    try:
        with open("faiss_store_openai.pkl", "rb") as f:
            faiss_index, summaries = pickle.load(f)
        answer, summary = answer_query(query, faiss_index, summaries)
        st.write("**Answer:**", answer)
        st.write("**Extracted Summary:**", summary)
    except FileNotFoundError:
        st.error("Please upload and process a PDF first.")
