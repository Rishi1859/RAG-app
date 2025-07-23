import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import streamlit as st
import os

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Load dataset
df = pd.read_csv("Training Dataset.csv")

# Convert rows to text
texts = []
for _, row in df.iterrows():
    row_dict = row.fillna("N/A").to_dict()
    content = ", ".join([f"{k}: {v}" for k, v in row_dict.items()])
    texts.append(Document(page_content=content))

# Vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(k=5)

# LLM & Chain
llm = OpenAI(temperature=0.3)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Streamlit UI
st.set_page_config(page_title="Loan Approval RAG Chatbot", layout="centered")
st.title("📊 Loan Approval RAG Chatbot")
query = st.text_input("Ask a question about the loan data:")

if query:
    with st.spinner("Searching..."):
        result = qa(query)
        st.success("Answer generated!")
        st.markdown("### 🧠 Answer")
        st.write(result["result"])
        st.markdown("### 🔍 Top Matching Entries")
        for doc in result["source_documents"]:
            st.markdown(f"- {doc.page_content[:300]}...")
