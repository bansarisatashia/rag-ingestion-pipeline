import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chunker import chunk_text, count_tokens  # noqa: E402
from embedder import HashingEmbedder  # noqa: E402
from loaders import load_folder  # noqa: E402
from pipeline import run  # noqa: E402
from store import InMemoryStore  # noqa: E402


def test_html_loader_strips_nav_and_footer():
    docs = {d.doc_id: d for d in load_folder(ROOT / "sample_docs")}
    html = docs["onboarding-guide.html"]
    assert html.title == "Data Platform Onboarding Guide"
    assert "Home | Docs" not in html.text and "Copyright" not in html.text
    assert "governance training" in html.text


def test_chunks_respect_max_tokens_and_overlap():
    text = " ".join(f"Sentence number {i} talks about pipelines." for i in range(300))
    chunks = chunk_text("doc", text, max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 5
    assert all(c.token_count <= 100 + 20 + 5 for c in chunks)
    first_tail = chunks[0].text.split()[-5:]
    assert chunks[1].text.split()[:15].count(first_tail[-1]) >= 1  # overlap carried over


def test_chunk_ids_are_deterministic():
    a = chunk_text("doc", "Alpha. Beta. Gamma.", max_tokens=50)
    b = chunk_text("doc", "Alpha. Beta. Gamma.", max_tokens=50)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_short_text_is_single_chunk():
    assert len(chunk_text("doc", "Just one line.", max_tokens=50)) == 1
    assert count_tokens("one two three") == 4


def test_incremental_runs(tmp_path):
    src = tmp_path / "docs"
    shutil.copytree(ROOT / "sample_docs", src)
    store, emb = InMemoryStore(), HashingEmbedder()

    first = run(str(src), store, emb, max_tokens=80)
    assert len(first.added) == 3 and first.chunks_written > 3

    second = run(str(src), store, emb, max_tokens=80)
    assert second.unchanged == 3 and second.chunks_written == 0  # no-op

    (src / "data-retention-policy.md").write_text("# Data Retention Policy\n\nLogs are kept for 30 days.")
    (src / "incident-response-runbook.md").unlink()
    third = run(str(src), store, emb, max_tokens=80)
    assert third.updated == ["data-retention-policy.md"]
    assert third.deleted == ["incident-response-runbook.md"]
    assert all(c.doc_id != "incident-response-runbook.md" for c, _ in store.chunks.values())


def test_retrieval_finds_relevant_document(tmp_path):
    store, emb = InMemoryStore(), HashingEmbedder()
    run(str(ROOT / "sample_docs"), store, emb, max_tokens=80)
    hits = store.search(emb.embed(["how long are transaction records retained for audit"])[0], k=3)
    assert hits[0]["doc_id"] == "data-retention-policy.md"
    hits = store.search(emb.embed(["page the on-call engineer for a SEV1 incident"])[0], k=3)
    assert hits[0]["doc_id"] == "incident-response-runbook.md"
