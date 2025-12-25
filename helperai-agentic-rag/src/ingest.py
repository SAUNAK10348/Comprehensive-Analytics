"""
Document ingestion for PDF, DOCX, TXT, and CSV with caching and validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pdfplumber
from docx import Document

from .chunking import chunk_text
from .config import CONFIG


def _hash_file(path: Path) -> str:
    """Return sha1 hash of a file for caching."""
    sha1 = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def _safe_path(path_str: str) -> Path:
    """Ensure path is safe (no traversal) and exists."""
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path_str}")
    if not str(path).startswith(str(Path(".").resolve())):
        raise ValueError("Unsafe path traversal detected.")
    return path


def _validate_file(path: Path) -> None:
    """Validate file extension and size."""
    if path.suffix.lower() not in CONFIG.allowed_ext:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > CONFIG.max_file_mb:
        raise ValueError(f"File {path.name} exceeds size limit of {CONFIG.max_file_mb} MB.")


def _cache_path(document_id: str) -> Path:
    cache_dir = Path(CONFIG.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{document_id}.json"


def _load_cache(document_id: str) -> List[Dict[str, str]] | None:
    cache_file = _cache_path(document_id)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _save_cache(document_id: str, data: List[Dict[str, str]]) -> None:
    cache_file = _cache_path(document_id)
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _parse_pdf(path: Path, document_id: str, timestamp: float) -> List[Tuple[str, Dict[str, str]]]:
    records: List[Tuple[str, Dict[str, str]]] = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            meta = {
                "document_id": document_id,
                "source": path.name,
                "page": str(page_num),
                "row": "",
                "ingested_at": str(timestamp),
            }
            records.append((text, meta))
    return records


def _parse_docx(path: Path, document_id: str, timestamp: float) -> List[Tuple[str, Dict[str, str]]]:
    doc = Document(path)
    records: List[Tuple[str, Dict[str, str]]] = []
    buffer = []
    for para in doc.paragraphs:
        if para.text.strip():
            buffer.append(para.text.strip())
    combined = "\n".join(buffer)
    meta = {
        "document_id": document_id,
        "source": path.name,
        "page": "",
        "row": "",
        "ingested_at": str(timestamp),
    }
    records.append((combined, meta))
    return records


def _parse_txt(path: Path, document_id: str, timestamp: float) -> List[Tuple[str, Dict[str, str]]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {
        "document_id": document_id,
        "source": path.name,
        "page": "",
        "row": "",
        "ingested_at": str(timestamp),
    }
    return [(text, meta)]


def _parse_csv(path: Path, document_id: str, timestamp: float) -> List[Tuple[str, Dict[str, str]]]:
    df = pd.read_csv(path)
    records: List[Tuple[str, Dict[str, str]]] = []
    for idx, row in df.iterrows():
        text = " ".join([f"{col}: {row[col]}" for col in df.columns])
        meta = {
            "document_id": document_id,
            "source": path.name,
            "page": "",
            "row": f"{idx}",
            "ingested_at": str(timestamp),
        }
        records.append((text, meta))
    return records


def ingest_document(path_str: str) -> List[Dict[str, str]]:
    """
    Ingest a single file and return chunked records with metadata.

    Args:
        path_str: file path

    Returns:
        List of chunk dicts with chunk_id, text, and metadata.
    """
    path = _safe_path(path_str)
    _validate_file(path)
    document_id = f"{path.stem}-{_hash_file(path)[:8]}"
    cached = _load_cache(document_id)
    if cached:
        return cached

    timestamp = time.time()
    if path.suffix.lower() == ".pdf":
        records = _parse_pdf(path, document_id, timestamp)
    elif path.suffix.lower() == ".docx":
        records = _parse_docx(path, document_id, timestamp)
    elif path.suffix.lower() == ".txt":
        records = _parse_txt(path, document_id, timestamp)
    elif path.suffix.lower() == ".csv":
        records = _parse_csv(path, document_id, timestamp)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    chunks: List[Dict[str, str]] = []
    for text, meta in records:
        chunks.extend(chunk_text(text, meta))
    _save_cache(document_id, chunks)
    return chunks


def ingest_multiple(paths: List[str]) -> List[Dict[str, str]]:
    """Ingest multiple files and aggregate chunks."""
    aggregated: List[Dict[str, str]] = []
    for p in paths:
        try:
            aggregated.extend(ingest_document(p))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to ingest {p}: {exc}")
    return aggregated
