"""
BM25 index wrapper.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .chunking import clean_text
from .config import CONFIG


class BM25Index:
    """Sparse retrieval using BM25."""

    def __init__(self) -> None:
        self.corpus: List[List[str]] = []
        self.doc_map: List[Dict[str, str]] = []
        self.model: BM25Okapi | None = None

    def build(self, chunks: List[Dict[str, str]]) -> None:
        """Build BM25 index from chunks."""
        self.doc_map = chunks
        tokenized = [clean_text(c["text"]).split() for c in chunks]
        self.corpus = tokenized
        self.model = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[Dict[str, str], float]]:
        """Search top_k documents."""
        if not self.model:
            return []
        k = top_k or CONFIG.top_k_sparse
        scores = self.model.get_scores(clean_text(query).split())
        indexed = list(enumerate(scores))
        ranked = sorted(indexed, key=lambda x: x[1], reverse=True)[:k]
        return [(self.doc_map[idx], float(score)) for idx, score in ranked]
