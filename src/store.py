"""Vector stores: Postgres + pgvector (production) and in-memory (tests)."""
from __future__ import annotations

import math
from typing import Protocol, Sequence

from chunker import Chunk


class VectorStore(Protocol):
    def doc_hashes(self) -> dict[str, str]: ...
    def replace_document(self, doc_id: str, title: str, content_hash: str,
                         chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def delete_document(self, doc_id: str) -> None: ...
    def search(self, query_vec: Sequence[float], k: int = 5) -> list[dict]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.chunks: dict[str, tuple[Chunk, list[float]]] = {}

    def doc_hashes(self) -> dict[str, str]:
        return {d: v["content_hash"] for d, v in self.docs.items()}

    def replace_document(self, doc_id, title, content_hash, chunks, vectors):
        self.delete_document(doc_id)
        self.docs[doc_id] = {"title": title, "content_hash": content_hash}
        for c, v in zip(chunks, vectors):
            self.chunks[c.chunk_id] = (c, list(v))

    def delete_document(self, doc_id):
        self.docs.pop(doc_id, None)
        for cid in [cid for cid, (c, _) in self.chunks.items() if c.doc_id == doc_id]:
            del self.chunks[cid]

    def search(self, query_vec, k=5):
        def cos(a, b):
            return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) or 1) * (math.sqrt(sum(y * y for y in b)) or 1))

        scored = sorted(((cos(query_vec, v), c) for c, v in self.chunks.values()), key=lambda t: t[0], reverse=True)
        return [
            {"doc_id": c.doc_id, "title": self.docs[c.doc_id]["title"], "ordinal": c.ordinal,
             "text": c.text, "score": round(s, 4)}
            for s, c in scored[:k]
        ]


class PgVectorStore:
    """Documents + chunks in Postgres with an HNSW cosine index (see sql/schema.sql)."""

    def __init__(self, dsn: str) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=False)
        register_vector(self.conn)

    def doc_hashes(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("select doc_id, content_hash from rag.documents")
            return dict(cur.fetchall())

    def replace_document(self, doc_id, title, content_hash, chunks, vectors):
        import numpy as np

        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("delete from rag.chunks where doc_id = %s", (doc_id,))
            cur.execute(
                """insert into rag.documents (doc_id, title, content_hash, updated_at)
                   values (%s, %s, %s, now())
                   on conflict (doc_id) do update
                   set title = excluded.title, content_hash = excluded.content_hash, updated_at = now()""",
                (doc_id, title, content_hash),
            )
            cur.executemany(
                """insert into rag.chunks (chunk_id, doc_id, ordinal, text, token_count, embedding)
                   values (%s, %s, %s, %s, %s, %s)""",
                [(c.chunk_id, c.doc_id, c.ordinal, c.text, c.token_count, np.array(v, dtype=np.float32))
                 for c, v in zip(chunks, vectors)],
            )

    def delete_document(self, doc_id):
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("delete from rag.documents where doc_id = %s", (doc_id,))  # chunks cascade

    def search(self, query_vec, k=5):
        import numpy as np

        with self.conn.cursor() as cur:
            cur.execute(
                """select c.doc_id, d.title, c.ordinal, c.text, 1 - (c.embedding <=> %s) as score
                   from rag.chunks c join rag.documents d using (doc_id)
                   order by c.embedding <=> %s limit %s""",
                (np.array(query_vec, dtype=np.float32), np.array(query_vec, dtype=np.float32), k),
            )
            cols = ["doc_id", "title", "ordinal", "text", "score"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
