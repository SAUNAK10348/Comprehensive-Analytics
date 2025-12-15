"""LLM helper for answering with context."""
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


def ask_llm(query: str, context: str, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    prompt = f"""
Use the information below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2
    )

    return completion.choices[0].message.content.strip()


__all__ = ["ask_llm"]
