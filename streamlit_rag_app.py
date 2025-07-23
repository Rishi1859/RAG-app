
import os
import faiss
import openai
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# === CONFIG ===
openai.api_key = os.getenv("OPENAI_API_KEY")
DATA_FOLDER = "data"
CHUNK_SIZE = 300
TOP_K = 3

# === INIT ===
st.set_page_config(page_title="RAG Q&A Chatbot", layout="wide")
st.title("📄 RAG Document Q&A Chatbot")

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedder()

# === UTILS ===
def load_and_chunk_pdfs(folder_path, chunk_size=300):
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(file_path)
                full_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
                for i in range(0, len(full_text), chunk_size):
                    chunk = full_text[i:i+chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)
            except Exception as e:
                st.warning(f"⚠️ Skipping {filename}: {e}")
    return chunks

def embed_texts(texts):
    return embed_model.encode(texts, convert_to_tensor=False)

def build_faiss_index(chunks):
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, embeddings

# === LOAD DOCS ===
os.makedirs(DATA_FOLDER, exist_ok=True)
chunks = load_and_chunk_pdfs(DATA_FOLDER, CHUNK_SIZE)

if not chunks:
    st.error("❌ No valid PDF chunks found. Please add readable PDF files to the 'data/' folder.")
    st.stop()

faiss_index, embeddings = build_faiss_index(chunks)

# === USER INPUT ===
question = st.text_input("Ask a question based on your uploaded documents:")

if st.button("Ask") and question.strip():
    try:
        q_embed = embed_texts([question])[0]
        D, I = faiss_index.search([q_embed], k=TOP_K)
        context = "\n---\n".join([chunks[i] for i in I[0]])

        prompt = f"Answer the question using the context:\n{context}\n\nQuestion: {question}\nAnswer:"
        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            st.success(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"❌ Error: {e}")
