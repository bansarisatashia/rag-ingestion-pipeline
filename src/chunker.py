"""Structure-aware recursive chunker with overlap.

Splits on the largest natural boundary that fits (sections, paragraphs,
sentences, then words) so chunks stay semantically coherent, then adds a
word overlap between neighbours so context isn't lost at the edges.
Sizes are in approximate tokens (1 token ~= 0.75 words), which keeps the
chunker dependency-free; swap `count_tokens` for tiktoken if needed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

SEPARATORS = [r"\n(?=#{1,6} )", r"\n\n", r"(?<=[.!?])\s+", r"\s+"]


def count_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / 0.75))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    token_count: int


def _split(text: str, max_tokens: int, level: int = 0) -> list[str]:
    if count_tokens(text) <= max_tokens or level >= len(SEPARATORS):
        return [text.strip()] if text.strip() else []
    pieces = [p for p in re.split(SEPARATORS[level], text) if p.strip()]
    out: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf}\n\n{piece}".strip() if buf and level < 2 else f"{buf} {piece}".strip()
        if count_tokens(candidate) <= max_tokens:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if count_tokens(piece) > max_tokens:
            out.extend(_split(piece, max_tokens, level + 1))
            buf = ""
        else:
            buf = piece.strip()
    if buf:
        out.append(buf)
    return out


def _with_overlap(parts: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0 or len(parts) < 2:
        return parts
    n_words = max(1, int(overlap_tokens * 0.75))
    out = [parts[0]]
    for prev, cur in zip(parts, parts[1:]):
        tail = " ".join(prev.split()[-n_words:])
        out.append(f"{tail} {cur}")
    return out


def chunk_text(doc_id: str, text: str, max_tokens: int = 300, overlap_tokens: int = 40) -> list[Chunk]:
    parts = _with_overlap(_split(text, max_tokens), overlap_tokens)
    chunks = []
    for i, part in enumerate(parts):
        cid = hashlib.sha1(f"{doc_id}:{i}:{part}".encode()).hexdigest()
        chunks.append(Chunk(chunk_id=cid, doc_id=doc_id, ordinal=i, text=part, token_count=count_tokens(part)))
    return chunks
