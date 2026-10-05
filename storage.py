import sqlite3
import numpy as np

def init_db():
    conn = sqlite3.connect("rag.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT,
            text TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    return conn

def save_chunk(conn, document_id, text, embedding):
    conn.execute(
        "INSERT INTO chunks (document_id, text, embedding) VALUES (?, ?, ?)",
        (document_id, text, embedding.tobytes())
    )
    conn.commit()

def get_all_chunks(conn):
    rows = conn.execute("SELECT document_id, text, embedding FROM chunks").fetchall()
    chunks = []
    for doc_id, text, emb_blob in rows:
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        chunks.append({"document_id": doc_id, "text": text, "embedding": embedding})
    return chunks