"""
Knowledge Graph construction and traversal with ontology validation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx

from .ontology import Ontology


def _simple_entity_candidates(text: str) -> List[str]:
    """Return capitalized word groups as crude entities."""
    candidates = re.findall(r"\b([A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]{2,})*)", text)
    return list({c.strip() for c in candidates})


def _match_entity_type(text: str, ontology: Ontology) -> str | None:
    lower = text.lower()
    for ent_type, info in ontology.entity_types.items():
        synonyms = [ent_type] + [s for s in info.get("synonyms", [])]
        if any(s.lower() in lower for s in synonyms):
            return ent_type
    return None


class KnowledgeGraphBuilder:
    """Builds and queries a lightweight ontology-constrained KG."""

    def __init__(self, ontology: Ontology | None = None) -> None:
        self.ontology = ontology or Ontology()
        self.graph = nx.Graph()

    def _add_entity(self, name: str, entity_type: str, source_chunk: str, aliases: List[str] | None = None) -> None:
        canonical = name.strip()
        if self.graph.has_node(canonical):
            node_data = self.graph.nodes[canonical]
            node_data["source_refs"].add(source_chunk)
            if aliases:
                node_data["aliases"].update(set(aliases))
            return
        self.graph.add_node(
            canonical,
            type=entity_type,
            aliases=set(aliases or []),
            source_refs={source_chunk},
        )

    def _add_relation(
        self,
        src: str,
        dst: str,
        relation_type: str,
        confidence: float,
        chunk_id: str,
        provenance: str,
    ) -> None:
        """Add relation if ontology allows; otherwise degrade confidence."""
        src_type = self.graph.nodes[src]["type"]
        dst_type = self.graph.nodes[dst]["type"]
        valid = self.ontology.allowed_relation(relation_type, src_type, dst_type)
        confidence_adj = confidence if valid else max(0.0, confidence - self.ontology.edge_confidence_penalty())
        if self.graph.has_edge(src, dst):
            data = self.graph[src][dst]
            data["relation_types"].add(relation_type)
            data["chunk_ids"].add(chunk_id)
            data["provenance"].add(provenance)
            data["confidence"] = max(data["confidence"], confidence_adj)
        else:
            self.graph.add_edge(
                src,
                dst,
                relation_types={relation_type},
                confidence=confidence_adj,
                chunk_ids={chunk_id},
                provenance={provenance},
            )

    def extract_entities(self, text: str, chunk_id: str) -> List[Tuple[str, str, float]]:
        """Heuristic entity detection."""
        entities: List[Tuple[str, str, float]] = []
        # pattern-based by synonyms
        for ent_type, info in self.ontology.entity_types.items():
            for syn in [ent_type] + info.get("synonyms", []):
                if syn.lower() in text.lower():
                    entities.append((syn.title(), ent_type, 0.65))
        # capitalized candidate names with type inference
        for cand in _simple_entity_candidates(text):
            inferred = _match_entity_type(text, self.ontology) or "Organization"
            entities.append((cand, inferred, 0.55))
        # deduplicate
        unique = {}
        for name, typ, conf in entities:
            key = name.lower()
            if key not in unique or unique[key][2] < conf:
                unique[key] = (name, typ, conf)
        return list(unique.values())

    def extract_relations(self, text: str, entities: List[Tuple[str, str, float]], chunk_id: str) -> List[Tuple[str, str, str, float]]:
        """Infer relations using simple patterns and co-occurrence fallback."""
        relations: List[Tuple[str, str, str, float]] = []
        lower = text.lower()
        patterns = [
            ("DEPENDS_ON", r"(.+?) depends on (.+?)"),
            ("OWNS", r"(.+?) owns (.+?)"),
            ("MITIGATES", r"(.+?) mitigates (.+?)"),
            ("VIOLATES", r"(.+?) violates (.+?)"),
            ("PRODUCES", r"(.+?) produces (.+?)"),
            ("CONSUMES", r"(.+?) uses (.+?)"),
        ]
        for rel_type, pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                src_text, dst_text = match.groups()
                src = src_text.strip().title()
                dst = dst_text.strip().title()
                relations.append((src, dst, rel_type, 0.8))
        # co-occurrence fallback
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                for j in range(i + 1, len(entities)):
                    src_name, _, _ = entities[i]
                    dst_name, _, _ = entities[j]
                    relations.append((src_name, dst_name, "RELATED_TO", 0.5 - self.ontology.co_occurrence_penalty()))
        return relations

    def build_from_chunks(self, chunks: List[Dict[str, str]]) -> nx.Graph:
        """Construct KG from chunked documents."""
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "")
            entities = self.extract_entities(text, chunk_id)
            for name, ent_type, _ in entities:
                self._add_entity(name, ent_type, chunk_id)
            relations = self.extract_relations(text, entities, chunk_id)
            for src, dst, rel_type, conf in relations:
                if src in self.graph.nodes and dst in self.graph.nodes:
                    self._add_relation(src, dst, rel_type, conf, chunk_id, chunk.get("source", ""))
        return self.graph

    def traverse(self, query: str, max_hops: int = 2, max_neighbors: int = 10) -> List[Dict[str, object]]:
        """Return ontology-filtered traversal evidence."""
        hits = []
        query_tokens = {q.lower() for q in re.findall(r"[A-Za-z0-9_-]+", query)}
        seed_nodes = [n for n in self.graph.nodes if any(tok in n.lower() for tok in query_tokens)]
        if not seed_nodes:
            # fallback to top-degree nodes
            degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
            seed_nodes = [n for n, _ in degrees[:3]]
        visited = set(seed_nodes)
        queue = list(seed_nodes)
        hops = 0
        while queue and hops <= max_hops:
            next_queue = []
            for node in queue:
                neighbors = list(self.graph.neighbors(node))[:max_neighbors]
                for neigh in neighbors:
                    edge = self.graph[node][neigh]
                    hits.append(
                        {
                            "source": node,
                            "target": neigh,
                            "relation_types": list(edge["relation_types"]),
                            "confidence": edge["confidence"],
                            "chunk_ids": list(edge["chunk_ids"]),
                            "provenance": list(edge["provenance"]),
                        }
                    )
                    if neigh not in visited:
                        visited.add(neigh)
                        next_queue.append(neigh)
            queue = next_queue
            hops += 1
        return hits

    def summary(self) -> Dict[str, int]:
        """Return node and edge counts."""
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}
