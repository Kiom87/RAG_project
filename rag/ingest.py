"""Load documents from the knowledge base and split them into chunks.

Supports Markdown/text files (.md, .txt) and PDF files (.pdf). PDFs let you
drop in your own company documents without changing any code.
"""

import os
from typing import List, Dict

from . import config


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(kb_dir: str = None) -> List[Dict]:
    """Walk the knowledge base directory and return a list of documents.

    Each document is a dict: {"text": str, "source": str}
    where ``source`` is the path relative to the knowledge base root.
    """
    kb_dir = kb_dir or config.KB_DIR
    docs: List[Dict] = []

    for root, _dirs, files in os.walk(kb_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext in (".md", ".txt"):
                    text = _read_text_file(path)
                elif ext == ".pdf":
                    text = _read_pdf_file(path)
                else:
                    continue
            except Exception as exc:  # don't let one bad file break ingestion
                print(f"[ingest] Skipping {path}: {exc}")
                continue

            text = text.strip()
            if text:
                source = os.path.relpath(path, kb_dir)
                docs.append({"text": text, "source": source})

    return docs


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

    # Split on blank lines first so chunks stay topically coherent.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # If a single paragraph is larger than chunk_size, hard-split it.
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    chunks.append(para[start : start + chunk_size])
                    start += chunk_size - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)

    # Add a small overlap between consecutive chunks for context continuity.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {chunks[i]}")
        chunks = overlapped

    return chunks


def build_chunks(kb_dir: str = None) -> List[Dict]:
    """Return a flat list of chunk dicts ready for embedding.

    Each chunk: {"id": str, "text": str, "source": str}
    """
    documents = load_documents(kb_dir)
    chunks: List[Dict] = []
    for doc in documents:
        for i, piece in enumerate(chunk_text(doc["text"])):
            chunks.append(
                {
                    "id": f"{doc['source']}::chunk-{i}",
                    "text": piece,
                    "source": doc["source"],
                }
            )
    return chunks
