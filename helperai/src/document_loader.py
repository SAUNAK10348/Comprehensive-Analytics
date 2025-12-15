"""Document loading helpers for PDF, DOCX, and TXT files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    logger.info("Loaded PDF %s with %d pages", path, len(pdf.pages))
    return text


def load_docx(path: str | Path) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    text = "\n".join(paragraphs)
    logger.info("Loaded DOCX %s with %d paragraphs", path, len(paragraphs))
    return text


def load_txt(path: str | Path, encoding: str = "utf-8") -> str:
    path = Path(path)
    text = path.read_text(encoding=encoding)
    logger.info("Loaded TXT %s (%d characters)", path, len(text))
    return text


def load_document(path: str | Path) -> str:
    """Load text content from supported document types."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".docx", ".doc"}:
        return load_docx(path)
    if suffix == ".txt":
        return load_txt(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


__all__ = ["load_pdf", "load_docx", "load_txt", "load_document", "chunk_text"]
