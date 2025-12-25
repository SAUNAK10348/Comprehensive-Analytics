"""Agentic orchestration for RAG + KG pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .hybrid_retriever import HybridRetriever
from .ingest import DocumentChunk
from .kg_builder import KnowledgeGraphBuilder
from .llm import LLMClient


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    source: str


class PlannerAgent:
    """Splits complex questions into targeted sub-questions."""

    def plan(self, query: str) -> List[str]:
        if " and " in query.lower():
            parts = [p.strip(" ?") for p in query.split(" and ") if p.strip()]
            return parts
        return [query]


class RetrieverAgent:
    """Retrieves evidence using hybrid search and KG traversal."""

    def __init__(self, retriever: HybridRetriever, kg: KnowledgeGraphBuilder) -> None:
        self.retriever = retriever
        self.kg = kg

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        hybrid_results = self.retriever.search(query, top_k=top_k)
        kg_entities = self.kg.traverse(query, k_hop=1)
        kg_chunks = set(self.kg.supporting_chunks(kg_entities))
        results: List[RetrievalResult] = []
        seen = set()
        for chunk, score, source in hybrid_results:
            results.append(RetrievalResult(chunk=chunk, score=score, source=source))
            seen.add(chunk.chunk_id)
        for chunk, score, source in hybrid_results:
            if chunk.chunk_id in kg_chunks and chunk.chunk_id not in seen:
                results.append(RetrievalResult(chunk=chunk, score=score + 0.1, source=source))
                seen.add(chunk.chunk_id)
        results = sorted(results, key=lambda x: x.score, reverse=True)
        return results[:top_k]


class VerifierAgent:
    """Checks that answers are grounded in evidence; triggers re-retrieval otherwise."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def verify(self, answer: str, evidence: List[DocumentChunk]) -> Tuple[bool, float]:
        score = self.llm.evaluate_faithfulness(answer, evidence)
        return score >= 0.4, score


class AgentOrchestrator:
    """Coordinates the planner, retriever, verifier, and LLM for QA."""

    def __init__(
        self,
        retriever: HybridRetriever,
        kg: KnowledgeGraphBuilder,
        llm: LLMClient,
        bm25_top_k: int = 5,
        faiss_top_k: int = 5,
        hybrid_top_k: int = 5,
    ) -> None:
        self.planner = PlannerAgent()
        self.retriever_agent = RetrieverAgent(retriever, kg)
        self.verifier = VerifierAgent(llm)
        self.llm = llm
        self.bm25_top_k = bm25_top_k
        self.faiss_top_k = faiss_top_k
        self.hybrid_top_k = hybrid_top_k

    def answer(self, query: str) -> Dict:
        start_time = time.time()
        sub_questions = self.planner.plan(query)
        evidence: List[DocumentChunk] = []
        retrievals: List[RetrievalResult] = []
        for sub in sub_questions:
            results = self.retriever_agent.retrieve(sub, top_k=self.hybrid_top_k)
            retrievals.extend(results)
            evidence.extend([r.chunk for r in results])
        # Deduplicate evidence
        seen_ids = set()
        unique_evidence: List[DocumentChunk] = []
        for chunk in evidence:
            if chunk.chunk_id not in seen_ids:
                unique_evidence.append(chunk)
                seen_ids.add(chunk.chunk_id)

        answer = self.llm.answer_with_fallback(query, unique_evidence)
        verified, verification_score = self.verifier.verify(answer, unique_evidence)
        latency = time.time() - start_time

        if not verified:
            # re-retrieve with broader scope
            extra_results = self.retriever_agent.retrieve(query, top_k=self.hybrid_top_k + 3)
            for r in extra_results:
                if r.chunk.chunk_id not in seen_ids:
                    unique_evidence.append(r.chunk)
                    seen_ids.add(r.chunk.chunk_id)
            answer = self.llm.answer_with_fallback(query, unique_evidence)
            verified, verification_score = self.verifier.verify(answer, unique_evidence)

        trace = {
            "sub_questions": sub_questions,
            "retrieved": [r.chunk.chunk_id for r in retrievals],
            "verification_score": verification_score,
            "latency_sec": latency,
            "coverage_score": self.llm.coverage_score(unique_evidence),
        }

        return {
            "answer": answer,
            "verified": verified,
            "evidence": unique_evidence,
            "trace": trace,
        }


__all__ = [
    "RetrievalResult",
    "PlannerAgent",
    "RetrieverAgent",
    "VerifierAgent",
    "AgentOrchestrator",
]
