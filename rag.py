import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from storage import get_all_chunks

load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def embed_texts(texts):
    """Convert a list of texts to embeddings using Groq."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float"
    )
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def embed_query(query):
    """Embed a single query."""
    return embed_texts([query])[0]


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunks(conn, query, top_k=3):
    """Find the most similar chunks to the query."""
    query_embedding = embed_query(query)
    all_chunks = get_all_chunks(conn)

    scored = []
    for chunk in all_chunks:
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((sim, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk["text"] for _, chunk in scored[:top_k]]


def generate_answer(query, context_chunks):
    """Generate a grounded answer using the retrieved context."""
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question based ONLY on the context below.
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content, context_chunks