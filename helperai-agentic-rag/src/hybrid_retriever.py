"""Hybrid retriever combining BM25 and FAISS results."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .bm25_index import BM25Index
from .faiss_index import FAISSIndex
from .ingest import DocumentChunk


class HybridRetriever:
    """Retrieves relevant chunks by merging sparse and dense scores."""

    def __init__(self, bm25_top_k: int = 5, faiss_top_k: int = 5) -> None:
        self.bm25_index = BM25Index()
        self.faiss_index = FAISSIndex()
        self.bm25_top_k = bm25_top_k
        self.faiss_top_k = faiss_top_k

    def build(self, chunks: List[DocumentChunk]) -> None:
        self.bm25_index.build(chunks)
        self.faiss_index.build(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float, str]]:
        bm25_results = self.bm25_index.search(query, top_k=self.bm25_top_k)
        dense_results = self.faiss_index.search(query, top_k=self.faiss_top_k)
        scored: Dict[str, Tuple[DocumentChunk, float]] = {}

        def _update(result_list: List[Tuple[DocumentChunk, float]], weight: float) -> None:
            for chunk, score in result_list:
                current = scored.get(chunk.chunk_id)
                blended = score * weight
                if current:
                    chunk_existing, existing_score = current
                    scored[chunk.chunk_id] = (chunk_existing, existing_score + blended)
                else:
                    scored[chunk.chunk_id] = (chunk, blended)

        _update(bm25_results, weight=1.0)
        _update(dense_results, weight=1.0)

        merged = sorted(scored.values(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(chunk, score, chunk.source) for chunk, score in merged]


__all__ = ["HybridRetriever"]
