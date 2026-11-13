"""Load documents from a folder into a normalized (doc_id, title, text, content_hash) form."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED = {".md", ".txt", ".html", ".htm"}


@dataclass(frozen=True)
class Document:
    doc_id: str          # stable id = path relative to the source root
    title: str
    text: str
    content_hash: str    # drives incremental re-ingestion


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(path: Path, root: Path) -> Document | None:
    if path.suffix.lower() not in SUPPORTED:
        return None
    raw = path.read_text(encoding="utf-8", errors="ignore")
    title = path.stem.replace("-", " ").replace("_", " ").title()
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(raw)
        text = "".join(parser.parts)
        title = parser.title.strip() or title
    else:
        text = raw
        m = re.search(r"^#\s+(.+)$", raw, flags=re.M)
        if m:
            title = m.group(1).strip()
    text = clean(text)
    if not text:
        return None
    return Document(
        doc_id=str(path.relative_to(root)),
        title=title,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def load_folder(root: str | Path) -> list[Document]:
    root = Path(root)
    docs = [load_file(p, root) for p in sorted(root.rglob("*")) if p.is_file()]
    return [d for d in docs if d]
