"""
Chunking utilities for semantic-safe splitting with overlap.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .config import CONFIG


def clean_text(text: str) -> str:
    """Normalize whitespace and guardrail against prompt injection markers."""
    text = re.sub(r"\s+", " ", text or "").strip()
    # Strip obvious instruction-style patterns to reduce injection risk.
    text = re.sub(r"(?i)(system:|assistant:|user:)", "", text)
    return text


def chunk_text(text: str, metadata: Dict[str, str], chunk_size: int | None = None, overlap: int | None = None) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks preserving metadata.

    Args:
        text: source text
        metadata: base metadata dict to copy
        chunk_size: optional override for chunk size
        overlap: optional override for overlap size

    Returns:
        List of chunk dicts with `chunk_id`, `text`, and metadata.
    """
    size = chunk_size or CONFIG.chunk_size
    ov = overlap or CONFIG.chunk_overlap
    normalized = clean_text(text)
    chunks: List[Dict[str, str]] = []
    start = 0
    idx = 0
    while start < len(normalized):
        end = start + size
        chunk_text_val = normalized[start:end]
        chunk_id = f"{metadata.get('document_id', 'doc')}-chunk-{idx}"
        entry = {**metadata, "chunk_id": chunk_id, "text": chunk_text_val}
        chunks.append(entry)
        start = end - ov
        idx += 1
    return chunks


def chunk_iterable(items: Iterable[Tuple[str, Dict[str, str]]]) -> List[Dict[str, str]]:
    """Helper for chunking multiple (text, metadata) tuples."""
    output: List[Dict[str, str]] = []
    for text, meta in items:
        output.extend(chunk_text(text, meta))
    return output
