from flask import Flask, request, jsonify
from utils import load_and_chunk_docs, build_faiss_index, embed_documents
import faiss
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # or use Claude/Gemini key

app = Flask(__name__)

# Step 1: Load and embed documents
chunks = load_and_chunk_docs("data")
index, embeddings = build_faiss_index(chunks)

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json["question"]
    query_embedding = embed_documents([query])[0]
    
    # Step 2: Retrieve top-k documents
    D, I = index.search([query_embedding], k=3)
    context = "\n".join([chunks[i] for i in I[0]])

    # Step 3: Generate answer using OpenAI (or Claude/Gemini if available)
    prompt = f"""Answer the question based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or use GPT-4 if available
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )

    return jsonify({"answer": response.choices[0].message.content.strip()})

if __name__ == "__main__":
    app.run(debug=True)
