CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.documents (
    doc_id        TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag.chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
    ordinal      INT  NOT NULL,
    text         TEXT NOT NULL,
    token_count  INT  NOT NULL,
    embedding    vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON rag.chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_doc_id ON rag.chunks (doc_id);
