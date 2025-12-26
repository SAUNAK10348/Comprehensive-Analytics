"""
Utility script to download and load a Llama model pipeline in Codespaces.

Usage:
    python -m src.load_llama_pipeline --model meta-llama/Llama-3-8b-instruct

The script performs:
    - Hugging Face login (reads HUGGINGFACE_TOKEN).
    - Local download of the model and tokenizer.
    - Creation of a transformers text-generation pipeline.
    - A quick smoke test generation on CPU (deterministic).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .llm import ensure_hf_login


def load_pipeline(model_name: str, trust_remote_code: bool = False):
    """Load tokenizer/model/pipeline for the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if not os.getenv("DISABLE_DEVICE_MAP") else None,
    )
    return gen


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Load a Llama model pipeline in Codespaces.")
    parser.add_argument("--model", default="meta-llama/Llama-3-8b-instruct", help="Model repo id")
    parser.add_argument("--prompt", default="Hello from Codespaces!", help="Smoke-test prompt")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code")
    ns = parser.parse_args(args=args)

    if not ensure_hf_login():
        print("Warning: HUGGINGFACE_TOKEN not provided; proceeding with public models only.", file=sys.stderr)
    gen = load_pipeline(ns.model, trust_remote_code=ns.trust_remote_code)
    output = gen(ns.prompt, max_new_tokens=32, do_sample=False)
    print("Sample output:", output[0]["generated_text"])


if __name__ == "__main__":
    main()
