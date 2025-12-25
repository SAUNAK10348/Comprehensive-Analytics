"""Chunking utilities for semantic-safe text splitting."""
from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional

from .ingest import DocumentChunk


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    base_metadata: Optional[Dict] = None,
    document_id: Optional[str] = None,
) -> List[DocumentChunk]:
    """Split text into overlapping chunks while preserving metadata.

    Args:
        text: The raw text to split.
        chunk_size: Target size of each chunk in characters.
        overlap: Overlap between consecutive chunks in characters.
        base_metadata: Shared metadata applied to all chunks from this text.
        document_id: Optional document identifier.

    Returns:
        List of DocumentChunk instances.
    """

    if not text:
        return []
    base_metadata = base_metadata or {}
    cleaned = " ".join(text.split())
    segments: List[str] = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        segments.append(cleaned[start:end])
        start = end - overlap
    chunks: List[DocumentChunk] = []
    total = len(segments)
    for idx, segment in enumerate(segments):
        chunk_id = f"{document_id or 'doc'}-chunk-{uuid.uuid4().hex[:8]}"
        metadata = {**base_metadata}
        metadata["chunk_index"] = idx
        metadata["chunk_count"] = total
        chunks.append(
            DocumentChunk(
                document_id=document_id or "unknown",
                chunk_id=chunk_id,
                text=segment,
                source=metadata.get("source", "unknown"),
                page=metadata.get("page"),
                row=metadata.get("row"),
                metadata=metadata,
            )
        )
    return chunks


__all__ = ["chunk_text"]
