from fastapi import FastAPI, HTTPException
from models import DocumentInput, QueryInput, QueryResponse
from storage import init_db, save_chunk
from rag import chunk_text, embed_texts, retrieve_relevant_chunks, generate_answer

app = FastAPI(title="RAG Chatbot API")
conn = init_db()


@app.post("/documents/add")
def add_document(doc: DocumentInput):
    """Ingest a document: chunk it, embed each chunk, and store in SQLite."""
    chunks = chunk_text(doc.text)
    embeddings = embed_texts(chunks)

    for text, emb in zip(chunks, embeddings):
        save_chunk(conn, doc.document_id, text, emb)

    return {"status": "ok", "chunks_added": len(chunks)}


@app.post("/chat/ask", response_model=QueryResponse)
def ask_question(q: QueryInput):
    """Retrieve relevant chunks and generate a grounded answer."""
    if not q.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    chunks = retrieve_relevant_chunks(conn, q.query)
    if not chunks:
        return QueryResponse(answer="No documents have been ingested yet.", sources=[])

    answer, sources = generate_answer(q.query, chunks)
    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "healthy"}