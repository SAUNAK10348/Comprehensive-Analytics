"""
Evaluation helpers for faithfulness and coverage.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set


def extract_citations(text: str) -> List[str]:
    """Extract citation markers like [doc-chunk-1]."""
    return re.findall(r"\[([\w.-]+)\]", text)


def evaluate_faithfulness(answer: str, evidence_chunks: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Compute simple faithfulness: percentage of sentences containing at least one valid citation.
    Valid citations must match chunk_ids present in evidence_chunks (not the concatenated text).
    """
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if s.strip()]
    if not sentences:
        return {"faithfulness": 0.0, "supported_sentences": 0, "total_sentences": 0}

    evidence_ids: Set[str] = {c.get("chunk_id", "") for c in evidence_chunks}
    supported = 0
    for sent in sentences:
        cites = extract_citations(sent)
        if any(c in evidence_ids for c in cites):
            supported += 1
    return {
        "faithfulness": supported / len(sentences),
        "supported_sentences": supported,
        "total_sentences": len(sentences),
    }


def coverage_score(answer: str, evidence_chunks: List[Dict[str, str]]) -> Dict[str, int]:
    """Return coverage metrics: unique chunks and sources cited."""
    citations = extract_citations(answer)
    cited_chunks = {c for c in citations}
    chunk_map = {chunk.get("chunk_id"): chunk for chunk in evidence_chunks}
    sources = {chunk_map[c]["source"] for c in cited_chunks if c in chunk_map}
    return {"unique_chunks_cited": len(cited_chunks), "unique_sources_cited": len(sources)}
