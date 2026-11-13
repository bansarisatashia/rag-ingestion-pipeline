"""Retrieval API: semantic search over the pgvector index (the 'R' in RAG)."""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, Query

from embedder import get_embedder
from store import PgVectorStore

app = FastAPI(title="RAG retrieval API")


@lru_cache
def _deps():
    return (
        get_embedder(os.getenv("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")),
        PgVectorStore(os.getenv("PG_DSN", "postgresql://rag:rag@postgres:5432/rag")),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)) -> dict:
    embedder, store = _deps()
    hits = store.search(embedder.embed([q])[0], k)
    return {"query": q, "results": hits}
