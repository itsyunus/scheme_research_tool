from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import tempfile

def process_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    loader = UnstructuredFileLoader(file_path)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(docs, embeddings)

    summaries = {}
    for doc in data:
        text = doc.page_content
        summaries["Summary"] = {
            "Eligibility": extract_section(text, "Eligibility"),
            "Benefits": extract_section(text, "Benefits"),
            "Documents": extract_section(text, "Documents"),
            "Application": extract_section(text, "Application")
        }
    return faiss_index, summaries

def answer_query(query, faiss_index, summaries):
    retriever = faiss_index.as_retriever()
    docs = retriever.get_relevant_documents(query)

    chain = load_qa_chain(ChatOpenAI(), chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)

    return response, summaries.get("Summary", "No summary available.")

def extract_section(text, keyword):
    for line in text.split("\n"):
        if keyword.lower() in line.lower():
            return line.strip()
    return "Not found"
