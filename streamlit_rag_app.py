import os
import faiss
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Constants
DATA_FOLDER = "docs"
CHUNK_SIZE = 500
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GENERATION_MODEL = "tiiuae/falcon-7b-instruct"  # or use OpenAI, Claude, etc.

# Load embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# Function to load and split PDFs into chunks
def load_and_chunk_pdfs(folder_path, chunk_size):
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, filename))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    for i in range(0, len(text), chunk_size):
                        chunks.append(text[i:i+chunk_size])
    return chunks

# Create FAISS index from chunks
def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Find top relevant chunks
def retrieve_top_chunks(question, chunks, embeddings, k=3):
    question_embedding = embed_model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    distances, indices = faiss_index.search(question_embedding, k)
    return [chunks[i] for i in indices[0]]

# Generate answer using a local or cloud model
def generate_answer(context, question):
    generator = pipeline("text-generation", model=GENERATION_MODEL)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    result = generator(prompt, max_length=300, do_sample=True)[0]['generated_text']
    return result.split("Answer:")[-1].strip()

# Streamlit UI
st.title("📚 RAG Q&A Chatbot")
question = st.text_input("Ask a question about your documents:")

# Process when button is clicked
if st.button("Submit") and question.strip():
    try:
        chunks = load_and_chunk_pdfs(DATA_FOLDER, CHUNK_SIZE)
        if not chunks:
            st.error("No text chunks found. Ensure your PDFs have extractable text.")
        else:
            faiss_index, embeddings = build_faiss_index(chunks)
            top_chunks = retrieve_top_chunks(question, chunks, embeddings)
            context = "\n".join(top_chunks)
            answer = generate_answer(context, question)
            st.success(answer)
    except Exception as e:
        st.error(f"❌ Error: {e}")
