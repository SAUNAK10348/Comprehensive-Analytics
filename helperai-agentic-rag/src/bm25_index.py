"""Sparse BM25 retrieval index."""
from __future__ import annotations

from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .ingest import DocumentChunk


class BM25Index:
    """BM25 index for sparse retrieval."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: List[DocumentChunk] = []

    def build(self, chunks: List[DocumentChunk]) -> None:
        self._chunks = chunks
        tokenized = [chunk.text.lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


__all__ = ["BM25Index"]
