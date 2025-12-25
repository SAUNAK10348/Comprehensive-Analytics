"""Dense retrieval using FAISS with sentence embeddings."""
from __future__ import annotations

import logging
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import DocumentChunk

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index backed by sentence-transformer embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: List[DocumentChunk] = []

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, chunks: List[DocumentChunk]) -> None:
        self._chunks = chunks
        if not chunks:
            self._index = None
            return
        model = self._ensure_model()
        embeddings = model.encode([c.text for c in chunks], convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = embeddings / norms
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self._index is None or not self._chunks:
            return []
        model = self._ensure_model()
        q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
        q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        scores, indices = self._index.search(q_emb, top_k)
        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self._chunks):
                results.append((self._chunks[idx], float(score)))
        return results


__all__ = ["FAISSIndex"]
