"""Embedding helper built on sentence-transformers and FAISS."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def build_index(chunks: List[str]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    logger.info("Built FAISS index with %d vectors", index.ntotal)
    return index, embeddings


def save_index(index: faiss.IndexFlatL2, chunks: List[str], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(directory / "index.faiss"))
    (directory / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved index and %d chunks to %s", len(chunks), directory)


def load_index(directory: str | Path) -> tuple[faiss.IndexFlatL2, List[str]]:
    directory = Path(directory)
    index = faiss.read_index(str(directory / "index.faiss"))
    chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    logger.info("Loaded index from %s containing %d chunks", directory, len(chunks))
    return index, chunks


def embed_query(query: str) -> np.ndarray:
    return model.encode([query])


__all__ = ["build_index", "save_index", "load_index", "embed_query", "model"]
