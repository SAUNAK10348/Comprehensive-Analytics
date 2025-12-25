"""LLM client with Ollama and extractive fallback."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import requests

from .ingest import DocumentChunk


class LLMClient:
    """Wrapper around Ollama with fallback extractive summarization."""

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.0"))

    def _ollama_available(self) -> bool:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self._ollama_available():
            return ""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "num_predict": max_tokens,
        }
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        if resp.status_code != 200:
            return ""
        output = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = line.decode("utf-8")
                if "response" in data:
                    # naive parsing, Ollama streams JSON per line
                    text = data.split("\"response\":\"")[-1].rsplit("\"", 1)[0]
                    output.append(text)
            except Exception:
                continue
        return "".join(output).strip()

    def answer_with_fallback(self, question: str, evidence: List[DocumentChunk]) -> str:
        context = "\n".join([f"[{chunk.chunk_id}] {chunk.text}" for chunk in evidence])
        prompt = (
            "You are a precise assistant. Answer the question using ONLY the provided context. "
            "Cite chunks inline using [chunk_id]. If unsure, say you do not know.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        generated = self.generate(prompt)
        if generated:
            return generated
        # Extractive fallback
        sentences = []
        for chunk in evidence:
            for sentence in chunk.text.split("."):
                if sentence.strip():
                    sentences.append((sentence.strip(), chunk.chunk_id))
        ranked = [s for s in sentences if any(word.lower() in s[0].lower() for word in question.split())]
        if not ranked:
            ranked = sentences
        top = ranked[:3]
        answer_parts = [f"{s} [{cid}]" for s, cid in top]
        return " ".join(answer_parts) if answer_parts else "I do not know."

    def evaluate_faithfulness(self, answer: str, evidence: List[DocumentChunk]) -> float:
        evidence_text = " ".join([c.text.lower() for c in evidence])
        cited = [seg.strip() for seg in answer.split() if seg.startswith("[") and seg.endswith("]")]
        hit = 0
        for token in cited:
            if token.strip("[]") in evidence_text:
                hit += 1
        if not cited:
            return 0.0
        return hit / len(cited)

    def coverage_score(self, evidence: List[DocumentChunk]) -> float:
        unique_sources = {c.source for c in evidence}
        return min(1.0, len(unique_sources) / 5)


__all__ = ["LLMClient"]
