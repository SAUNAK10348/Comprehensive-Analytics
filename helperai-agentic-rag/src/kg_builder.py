"""Knowledge graph construction utilities."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx

from .ingest import DocumentChunk


def extract_entities(text: str) -> List[str]:
    """Extract lightweight entities using capitalization heuristics and nouns."""
    candidates = re.findall(r"[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*", text)
    cleaned = [c.strip() for c in candidates if len(c.strip()) > 2]
    return cleaned


def normalize_entity(entity: str) -> str:
    return entity.lower().strip()


class KnowledgeGraphBuilder:
    """Builds and queries a lightweight knowledge graph."""

    def __init__(self) -> None:
        self.graph = nx.Graph()

    def add_documents(self, chunks: Iterable[DocumentChunk]) -> None:
        for chunk in chunks:
            entities = extract_entities(chunk.text)
            normalized = [normalize_entity(e) for e in entities]
            for ent in normalized:
                if not self.graph.has_node(ent):
                    self.graph.add_node(ent, label=ent)
            for i, src in enumerate(normalized):
                for tgt in normalized[i + 1 :]:
                    if src == tgt:
                        continue
                    if self.graph.has_edge(src, tgt):
                        self.graph[src][tgt]["weight"] += 1
                    else:
                        self.graph.add_edge(src, tgt, weight=1)
            # attach chunk reference
            for ent in normalized:
                chunks_for_ent = self.graph.nodes[ent].setdefault("chunks", [])
                chunks_for_ent.append(chunk.chunk_id)

    def traverse(self, query: str, k_hop: int = 1) -> Set[str]:
        entities = [normalize_entity(e) for e in extract_entities(query)]
        visited: Set[str] = set()
        for ent in entities:
            if ent not in self.graph:
                continue
            visited.add(ent)
            frontier = {ent}
            for _ in range(k_hop):
                next_frontier: Set[str] = set()
                for node in frontier:
                    neighbors = set(self.graph.neighbors(node))
                    next_frontier.update(neighbors)
                visited.update(next_frontier)
                frontier = next_frontier
        return visited

    def supporting_chunks(self, entity_set: Set[str]) -> List[str]:
        chunk_ids: Set[str] = set()
        for ent in entity_set:
            data = self.graph.nodes.get(ent, {})
            for chunk_id in data.get("chunks", []):
                chunk_ids.add(chunk_id)
        return list(chunk_ids)

    def summary(self, limit: int = 10) -> List[Tuple[str, int]]:
        degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
        return degrees[:limit]


__all__ = ["KnowledgeGraphBuilder", "extract_entities", "normalize_entity"]
