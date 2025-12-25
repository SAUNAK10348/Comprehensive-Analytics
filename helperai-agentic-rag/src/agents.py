"""
Agentic workflow: planner, retriever, verifier, and orchestrator.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Dict, List, Tuple

from .config import CONFIG
from .eval_utils import evaluate_faithfulness, extract_citations
from .hybrid_retriever import HybridRetriever
from .kg_builder import KnowledgeGraphBuilder
from .llm import answer_question
from .ontology import Ontology


class PlannerAgent:
    """Decomposes queries and infers ontology lens."""

    def __init__(self, ontology: Ontology) -> None:
        self.ontology = ontology

    def plan(self, query: str) -> Dict[str, object]:
        lower = query.lower()
        if any(k in lower for k in ["policy", "compliance", "standard"]):
            intent = "policy"
        elif any(k in lower for k in ["metric", "kpi", "measure"]):
            intent = "metric"
        elif any(k in lower for k in ["depend", "integration", "architecture", "system"]):
            intent = "dependency"
        elif any(k in lower for k in ["risk", "incident", "outage"]):
            intent = "risk"
        else:
            intent = "default"
        subquestions = [query]
        return {"intent": intent, "subquestions": subquestions, "ontology_lens": self.ontology.bias_entities_for_intent(intent)}


class RetrieverAgent:
    """Retrieves evidence from text indexes and KG traversal."""

    def __init__(self, retriever: HybridRetriever, kg_builder: KnowledgeGraphBuilder) -> None:
        self.retriever = retriever
        self.kg_builder = kg_builder

    def retrieve(self, query: str, ontology_lens: List[str]) -> Dict[str, object]:
        text_results = self.retriever.retrieve(query, preferred_types=ontology_lens)
        chunks = [r["chunk"] for r in text_results]
        kg_hits = self.kg_builder.traverse(query, max_hops=CONFIG.kg_max_hops, max_neighbors=CONFIG.kg_max_neighbors)
        return {"chunks": chunks, "kg": kg_hits}


class VerifierAgent:
    """Validates faithfulness and flags prompt-injection attempts."""

    def verify(self, answer: str, evidence: List[Dict[str, str]]) -> Dict[str, object]:
        faithfulness = evaluate_faithfulness(answer, evidence)
        suspicious = any(x in answer.lower() for x in ["ignore previous", "disregard earlier", "follow these instructions"])
        missing_citations = len(extract_citations(answer)) == 0
        verdict = faithfulness["faithfulness"] >= 0.5 and not suspicious and not missing_citations
        return {"verdict": verdict, "faithfulness": faithfulness, "suspicious": suspicious, "missing_citations": missing_citations}


class AgentOrchestrator:
    """Controller that runs planner → retriever → verifier."""

    def __init__(self, retriever: HybridRetriever, kg_builder: KnowledgeGraphBuilder, ontology: Ontology | None = None) -> None:
        self.ontology = ontology or Ontology()
        self.planner = PlannerAgent(self.ontology)
        self.retriever_agent = RetrieverAgent(retriever, kg_builder)
        self.verifier = VerifierAgent()

    @staticmethod
    def _redact(text: str) -> str:
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
        text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[redacted-phone]", text)
        return text

    def _log(self, request_id: str, event: str, payload: Dict[str, object]) -> None:
        record = {"request_id": request_id, "event": event, **payload}
        print(json.dumps(record))

    def run(self, query: str, max_turns: int = 2) -> Dict[str, object]:
        request_id = str(uuid.uuid4())
        plan = self.planner.plan(query)
        intent = plan["intent"]
        ontology_lens = plan["ontology_lens"]
        all_chunks: List[Dict[str, str]] = []
        kg_evidence: List[Dict[str, object]] = []
        answer_payload: Dict[str, object] = {}
        timings: Dict[str, float] = {}

        self._log(request_id, "plan", {"intent": intent, "subquestions": plan["subquestions"]})

        for turn in range(max_turns):
            t0 = time.time()
            retrieved = self.retriever_agent.retrieve(query, ontology_lens)
            timings["retrieve"] = time.time() - t0
            all_chunks = retrieved["chunks"]
            kg_evidence = retrieved["kg"]

            llm_resp = answer_question(query, all_chunks, ontology_lens)
            timings["llm"] = llm_resp["latency"]
            answer_text = llm_resp["answer"]

            verification = self.verifier.verify(answer_text, all_chunks)
            self._log(
                request_id,
                "verification",
                {
                    "turn": turn,
                    "faithfulness": verification.get("faithfulness", {}),
                    "verdict": verification.get("verdict"),
                },
            )
            if verification["verdict"] or turn == max_turns - 1:
                answer_payload = {"answer": answer_text, "verification": verification}
                break
            # adjust query slightly using intent for second pass
            query = f"{query} Consider ontology intent {intent} and include policy/system specifics."

        return {
            "plan": plan,
            "answer": answer_payload.get("answer", ""),
            "verification": answer_payload.get("verification", {}),
            "evidence_chunks": all_chunks,
            "kg_evidence": kg_evidence,
            "timings": timings,
            "request_id": request_id,
        }
