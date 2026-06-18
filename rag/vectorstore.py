"""Embeddings + ChromaDB vector store.

Embeddings are computed locally with sentence-transformers (free, no API key).
ChromaDB stores the vectors and performs similarity search. We always supply
our own embeddings, so Chroma never needs to download its default model.
"""

from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from . import config


class Embedder:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = None):
        self.model = SentenceTransformer(model_name or config.EMBEDDING_MODEL)

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()


class VectorStore:
    """ChromaDB-backed vector store with cosine similarity."""

    def __init__(self):
        self.embedder = Embedder()
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def index(self, chunks: List[Dict], batch_size: int = 64) -> None:
        """Embed and store chunks. Idempotent: clears the collection first."""
        # Reset so re-running never creates duplicates.
        existing = self.collection.get()
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedder.encode([c["text"] for c in batch])
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=embeddings,
                metadatas=[{"source": c["source"]} for c in batch],
            )

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Return the most relevant chunks for a query."""
        top_k = top_k or config.TOP_K
        query_embedding = self.embedder.encode([query])
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        hits: List[Dict] = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(documents, metadatas, distances):
            hits.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "score": round(1 - dist, 3),  # cosine distance -> similarity
                }
            )
        return hits
