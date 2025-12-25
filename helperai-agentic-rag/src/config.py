"""
Configuration management for the Agentic RAG system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict


def _get_env(key: str, default: Any) -> Any:
    """Fetch environment variable with fallback and type casting."""
    val = os.getenv(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(val)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(val)
        except ValueError:
            return default
    return val


@dataclass
class AppConfig:
    """Central application configuration with env overrides."""

    # Paths
    data_dir: str = _get_env("DATA_DIR", "data")
    cache_dir: str = _get_env("CACHE_DIR", "data/cache")
    embeddings_dir: str = _get_env("EMBEDDINGS_DIR", "embeddings")
    ontology_path: str = _get_env("ONTOLOGY_PATH", "ontology/enterprise_ontology.json")

    # Ingestion
    max_file_mb: int = _get_env("MAX_FILE_MB", 20)
    allowed_ext: tuple[str, ...] = (".pdf", ".docx", ".txt", ".csv")

    # Chunking
    chunk_size: int = _get_env("CHUNK_SIZE", 1000)
    chunk_overlap: int = _get_env("CHUNK_OVERLAP", 200)

    # Retrieval weights
    top_k_sparse: int = _get_env("TOP_K_SPARSE", 10)
    top_k_dense: int = _get_env("TOP_K_DENSE", 10)
    weight_sparse: float = _get_env("WEIGHT_SPARSE", 0.55)
    weight_dense: float = _get_env("WEIGHT_DENSE", 0.45)

    # KG traversal
    kg_max_hops: int = _get_env("KG_MAX_HOPS", 2)
    kg_max_neighbors: int = _get_env("KG_MAX_NEIGHBORS", 10)

    # LLM
    ollama_model: str = _get_env("OLLAMA_MODEL", "llama3")
    temperature: float = _get_env("LLM_TEMPERATURE", 0.0)
    max_tokens: int = _get_env("LLM_MAX_TOKENS", 512)
    hf_token_env: str = _get_env("HUGGINGFACE_TOKEN", "")

    # Safety
    redaction: bool = _get_env("ENABLE_REDACTION", True)

    # Logging
    log_level: str = _get_env("LOG_LEVEL", "INFO")

    # Misc
    seed: int = _get_env("GLOBAL_SEED", 42)

    def as_dict(self) -> Dict[str, Any]:
        """Return config as dict."""
        return self.__dict__.copy()


CONFIG = AppConfig()
