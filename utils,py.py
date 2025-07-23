from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_documents(docs):
    return model.encode(docs, convert_to_tensor=False)

def build_faiss_index(text_chunks):
    embeddings = embed_documents(text_chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)
    return index, embeddings

def save_index(index, file_path="vectorstore/index.faiss"):
    faiss.write_index(index, file_path)

def load_index(file_path="vectorstore/index.faiss"):
    return faiss.read_index(file_path)

from PyPDF2 import PdfReader
import os

def load_and_chunk_docs(folder_path, chunk_size=300):
    chunks = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i+chunk_size])
    return chunks
