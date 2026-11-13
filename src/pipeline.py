"""Incremental RAG ingestion: load -> diff by content hash -> chunk -> embed -> upsert.

Only new or changed documents are re-chunked and re-embedded; documents removed
from the source are deleted from the index. Re-running on unchanged input is a no-op.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field

from chunker import chunk_text
from embedder import Embedder, get_embedder
from loaders import load_folder
from store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-ingest")


@dataclass
class RunStats:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: int = 0
    chunks_written: int = 0


def run(source_dir: str, store: VectorStore, embedder: Embedder,
        max_tokens: int = 300, overlap_tokens: int = 40) -> RunStats:
    stats = RunStats()
    docs = load_folder(source_dir)
    existing = store.doc_hashes()
    current_ids = {d.doc_id for d in docs}

    for doc in docs:
        old_hash = existing.get(doc.doc_id)
        if old_hash == doc.content_hash:
            stats.unchanged += 1
            continue
        chunks = chunk_text(doc.doc_id, doc.text, max_tokens, overlap_tokens)
        vectors = embedder.embed([f"{doc.title}\n\n{c.text}" for c in chunks])
        store.replace_document(doc.doc_id, doc.title, doc.content_hash, chunks, vectors)
        (stats.updated if old_hash else stats.added).append(doc.doc_id)
        stats.chunks_written += len(chunks)

    for doc_id in sorted(set(existing) - current_ids):
        store.delete_document(doc_id)
        stats.deleted.append(doc_id)

    log.info(
        "added=%d updated=%d deleted=%d unchanged=%d chunks=%d",
        len(stats.added), len(stats.updated), len(stats.deleted), stats.unchanged, stats.chunks_written,
    )
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="sample_docs")
    p.add_argument("--dsn", default=os.getenv("PG_DSN", "postgresql://rag:rag@localhost:5433/rag"))
    p.add_argument("--embedder", default=os.getenv("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"))
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--overlap", type=int, default=40)
    args = p.parse_args()

    from store import PgVectorStore

    run(args.source, PgVectorStore(args.dsn), get_embedder(args.embedder), args.max_tokens, args.overlap)


if __name__ == "__main__":
    main()
