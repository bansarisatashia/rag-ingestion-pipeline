# RAG Data Pipeline: Incremental Document Ingestion into pgvector

The data-engineering half of a retrieval-augmented generation (RAG) system. It turns a folder of documents (Markdown, text, HTML) into a searchable vector index in Postgres + pgvector. Only new or changed documents are re-processed, deleted documents are removed from the index, and a FastAPI endpoint serves semantic search for an LLM application.

> Personal portfolio project. The sample documents describe a fictional company.

## Architecture

```
 sample_docs/ (md, txt, html)
        │
        ▼
 ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌──────────────────────┐
 │ Load + clean │ → │ Diff by SHA-256 │ → │ Structure-aware  │ → │ Embed          │ → │ Postgres + pgvector  │
 │ strip HTML   │   │ content hash    │   │ chunking with    │   │ MiniLM-L6-v2   │   │ HNSW cosine index    │
 │ nav / footer │   │ new / changed / │   │ overlap          │   │ (384 dims)     │   │ one txn per document │
 └──────────────┘   │ deleted         │   └──────────────────┘   └────────────────┘   └──────────┬───────────┘
                    └─────────────────┘                                                         │
          Airflow @hourly ─ re-runs the pipeline; unchanged input is a no-op           FastAPI /search?q=…
```

## Design choices

- **Incremental by content hash:** each document's SHA-256 is stored. Unchanged documents are skipped, so hourly runs cost almost nothing, and embedding spend tracks real change.
- **Deletes propagate:** documents removed at the source are removed from the index, so the LLM never cites stale content.
- **Atomic per-document replace:** old chunks are deleted and new ones inserted in a single transaction. Readers never see a half-updated document.
- **Structure-aware chunking:** text is split on headings, then paragraphs, then sentences, then words, with token overlap between neighbouring chunks. Chunk IDs are deterministic.
- **Pluggable embeddings:** sentence-transformers by default. A deterministic hashing embedder lets tests and CI run without downloading a model.
- **Fast retrieval:** an HNSW index on `vector_cosine_ops` in pgvector.

## Run it

Requirements: Docker and Docker Compose.

```bash
docker compose up --build
curl "http://localhost:8000/search?q=how+long+do+we+keep+transaction+records&k=3"
```

Run the ingestion locally against the Compose database:

```bash
pip install -r requirements.txt
python src/pipeline.py --source sample_docs --dsn postgresql://rag:rag@localhost:5433/rag
```

Run it a second time and the log shows `unchanged=3 chunks=0`. Edit or delete a file and only that document is re-processed.

## Tests

```bash
pytest -q
```

The tests cover HTML cleaning, chunk size and overlap, deterministic chunk IDs, the full incremental lifecycle (add → no-op → update → delete), and retrieval returning the right document for a question.

## Project structure

```
src/loaders.py     load + clean md/txt/html, content hashing
src/chunker.py     recursive structure-aware chunker with overlap
src/embedder.py    sentence-transformers + deterministic hashing embedder
src/store.py       pgvector store (HNSW) + in-memory store
src/pipeline.py    incremental ingestion run
src/api.py         FastAPI retrieval endpoint
sql/schema.sql     documents/chunks tables, HNSW index
dags/rag_ingest_dag.py   hourly Airflow DAG
```

## Next steps

- Add hybrid search (pgvector + Postgres full-text with reciprocal rank fusion).
- Track retrieval quality with a small labeled question set (recall@k, MRR) in CI.
- Ingest PDFs and Confluence/SharePoint pages through connectors.

## Tech

Python · PostgreSQL · pgvector · sentence-transformers · FastAPI · Apache Airflow · Docker · pytest
