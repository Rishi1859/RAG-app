import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set your OpenAI key if using OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"

def load_docs(uploaded_file):
    path = f"temp_{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = TextLoader(path)
    return loader.load()

def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3)
    
    prompt = PromptTemplate.from_template(
        "Answer the question using the below context:\n{context}\n\nQuestion: {question}"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Streamlit UI
st.set_page_config(page_title="📚 RAG Q&A Chatbot", layout="wide")
st.title("📚 RAG Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a document (TXT or PDF)", type=["txt", "pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        docs = load_docs(uploaded_file)
        vectorstore = create_vector_store(docs)
        qa_chain = create_rag_chain(vectorstore)
    st.success("Document indexed successfully! Ask your question below.")

    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": question})
            st.markdown("### 🧠 Answer:")
            st.markdown(result["result"])
            with st.expander("🔍 Sources"):
                for doc in result["source_documents"]:
                    st.markdown(f"• {doc.page_content[:300]}...")

else:
    st.info("Please upload a document to get started.")
