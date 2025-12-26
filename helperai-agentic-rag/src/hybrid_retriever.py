"""
Hybrid retrieval combining BM25 and FAISS.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict

from .bm25_index import BM25Index
from .config import CONFIG
from .faiss_index import DenseFaissIndex


class HybridResult(TypedDict):
    chunk: Dict[str, str]
    bm25: float
    dense: float
    combined_score: float


class HybridRetriever:
    """Merge sparse and dense retrieval results."""

    def __init__(self) -> None:
        self.bm25 = BM25Index()
        self.faiss = DenseFaissIndex()

    def build(self, chunks: List[Dict[str, str]]) -> None:
        self.bm25.build(chunks)
        self.faiss.build(chunks)

    def retrieve(self, query: str, preferred_types: List[str] | None = None) -> List[HybridResult]:
        """Retrieve and merge results."""
        sparse = self.bm25.search(query, CONFIG.top_k_sparse)
        dense = self.faiss.search(query, CONFIG.top_k_dense)
        combined: Dict[str, HybridResult] = {}
        for doc, score in sparse:
            cid = doc["chunk_id"]
            combined[cid] = {
                "chunk": doc,
                "bm25": float(score),
                "dense": 0.0,
                "combined_score": CONFIG.weight_sparse * float(score),
            }
        for doc, score in dense:
            cid = doc["chunk_id"]
            if cid not in combined:
                combined[cid] = {
                    "chunk": doc,
                    "bm25": 0.0,
                    "dense": float(score),
                    "combined_score": CONFIG.weight_dense * float(score),
                }
            else:
                combined[cid]["dense"] = float(score)
                combined[cid]["combined_score"] += CONFIG.weight_dense * float(score)

        results = list(combined.values())
        # Optional ontology bias: boost if chunk metadata hints at preferred types
        if preferred_types:
            for item in results:
                meta_text = (item["chunk"].get("text", "") + " " + item["chunk"].get("source", "")).lower()
                if any(t.lower() in meta_text for t in preferred_types):
                    item["combined_score"] *= 1.05

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results
