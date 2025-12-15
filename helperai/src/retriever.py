"""Semantic retriever using FAISS."""
from __future__ import annotations

from typing import List

import faiss

from .embedder import embed_query


def retrieve(query: str, index: faiss.IndexFlatL2, chunks: List[str], k: int = 5) -> List[str]:
    q_emb = embed_query(query)
    distances, ids = index.search(q_emb, k)
    valid_ids = [i for i in ids[0] if i < len(chunks) and i >= 0]
    return [chunks[i] for i in valid_ids]


__all__ = ["retrieve"]
