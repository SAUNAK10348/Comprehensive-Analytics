"""
Dense retrieval with FAISS and sentence-transformer embeddings (with TF-IDF fallback).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import faiss
import numpy as np
from huggingface_hub import login
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import clean_text
from .config import CONFIG

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None  # type: ignore


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseFaissIndex:
    """Dense retrieval index."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self.model_name = model_name
        self.embedder = None
        self.index: faiss.IndexFlatIP | None = None
        self.doc_map: List[Dict[str, str]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.use_tfidf_fallback = False
        random.seed(CONFIG.seed)
        np.random.seed(CONFIG.seed)

    def _ensure_model(self) -> None:
        if self.embedder or self.use_tfidf_fallback:
            return
        if CONFIG.hf_token_env:
            try:
                login(token=CONFIG.hf_token_env, add_to_git_credential=False)
            except Exception:  # noqa: BLE001
                pass
        if SentenceTransformer:
            try:
                self.embedder = SentenceTransformer(self.model_name)
                return
            except Exception:  # noqa: BLE001
                self.embedder = None
        self.use_tfidf_fallback = True
        self.vectorizer = TfidfVectorizer(max_features=2048)

    def _encode(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        if self.embedder:
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            return np.asarray(embeddings, dtype=np.float32)
        assert self.vectorizer is not None
        if not hasattr(self.vectorizer, "vocabulary_"):
            self.vectorizer.fit(texts)
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)

    def build(self, chunks: List[Dict[str, str]]) -> None:
        """Build FAISS index from chunks."""
        self.doc_map = chunks
        texts = [clean_text(c["text"]) for c in chunks]
        vectors = self._encode(texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[Dict[str, str], float]]:
        """Search top_k vectors."""
        if self.index is None:
            return []
        k = top_k or CONFIG.top_k_dense
        q_vec = self._encode([clean_text(query)])
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, k)
        results: List[Tuple[Dict[str, str], float]] = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx < 0 or idx >= len(self.doc_map):
                continue
            results.append((self.doc_map[idx], float(score)))
        return results
