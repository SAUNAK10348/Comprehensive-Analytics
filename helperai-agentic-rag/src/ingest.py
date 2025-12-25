"""Document ingestion utilities for multi-format inputs.

This module normalizes PDF, DOCX, TXT, and CSV files into text chunks with
rich metadata for downstream retrieval and graph building.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import PyPDF2
from docx import Document

from .chunking import chunk_text


@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata for retrieval and tracing."""

    document_id: str
    chunk_id: str
    text: str
    source: str
    page: Optional[int] = None
    row: Optional[int] = None
    metadata: Optional[Dict] = None


class DocumentIngestor:
    """Ingests documents from disk into normalized text chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest(self, file_path: Path) -> List[DocumentChunk]:
        """Ingest a file into document chunks with metadata.

        Args:
            file_path: Path to the input document.

        Returns:
            List of DocumentChunk instances.
        """

        suffix = file_path.suffix.lower()
        document_id = str(uuid.uuid4())
        if suffix == ".pdf":
            texts = self._read_pdf(file_path)
        elif suffix in {".docx", ".doc"}:
            texts = self._read_docx(file_path)
        elif suffix == ".txt":
            texts = self._read_txt(file_path)
        elif suffix == ".csv":
            texts = self._read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        chunks: List[DocumentChunk] = []
        for entry in texts:
            raw_text = entry["text"]
            meta = {k: v for k, v in entry.items() if k != "text"}
            meta.setdefault("source", file_path.name)
            per_chunk = chunk_text(
                raw_text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                base_metadata=meta,
                document_id=document_id,
            )
            chunks.extend(per_chunk)
        return chunks

    def _read_pdf(self, file_path: Path) -> List[Dict]:
        results: List[Dict] = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                normalized = text.replace("\u0000", " ").strip()
                if normalized:
                    results.append({"text": normalized, "page": idx, "source": file_path.name})
        return results

    def _read_docx(self, file_path: Path) -> List[Dict]:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        content = "\n".join(paragraphs)
        return [{"text": content, "source": file_path.name}]

    def _read_txt(self, file_path: Path) -> List[Dict]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [{"text": text, "source": file_path.name}]

    def _read_csv(self, file_path: Path) -> List[Dict]:
        df = pd.read_csv(file_path)
        results: List[Dict] = []
        for idx, row in df.iterrows():
            row_text = " ".join([f"{col}: {row[col]}" for col in df.columns])
            results.append({"text": row_text, "row": int(idx), "source": file_path.name})
        return results


__all__ = ["DocumentChunk", "DocumentIngestor"]
