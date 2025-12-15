"""LLM helper for answering with context using a local Ollama model."""
from __future__ import annotations

import ollama


DEFAULT_MODEL = "llama3:instruct"


def ask_llm(query: str, context: str, model: str = DEFAULT_MODEL) -> str:
    """Send a contextual prompt to the Ollama chat API and return the response."""

    prompt = f"""
Use the information below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    completion = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )

    return completion.message.get("content", "").strip()


__all__ = ["ask_llm", "DEFAULT_MODEL"]
