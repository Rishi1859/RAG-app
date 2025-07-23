import os
import faiss
import openai
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# === CONFIGURATION ===
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly: "sk-..."
DATA_FOLDER = "data"
CHUNK_SIZE = 300
TOP_K = 3

# === INIT APP ===
app = Flask(__name__)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# === LOAD & CHUNK DOCUMENTS ===
def load_and_chunk_pdfs(folder_path, chunk_size=300):
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, filename))
            full_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i+chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
    return chunks

# === EMBEDDING & FAISS INDEX ===
def embed_texts(texts):
    return embed_model.encode(texts, convert_to_tensor=False)

def build_faiss_index(chunks):
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, embeddings

# === LOAD DOCUMENTS AND BUILD INDEX ===
print("Loading documents and building vector index...")
chunks = load_and_chunk_pdfs(DATA_FOLDER, CHUNK_SIZE)
faiss_index, embeddings = build_faiss_index(chunks)
print(f"Loaded {len(chunks)} chunks.")

# === RAG QA ENDPOINT ===
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Embed the user question
    question_embedding = embed_texts([user_question])[0]

    # Retrieve top-k relevant chunks
    D, I = faiss_index.search([question_embedding], k=TOP_K)
    context = "\n---\n".join([chunks[i] for i in I[0]])

    # Build the prompt
    prompt = f"""You are a helpful assistant. Use the context to answer the question.
Context:
{context}

Question: {user_question}
Answer:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer})

# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True)
