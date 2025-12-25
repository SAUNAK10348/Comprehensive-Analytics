# HelperAI Agentic RAG + Knowledge Graph

This project implements a fully local, production-style agentic Retrieval-Augmented Generation (RAG) system with a lightweight knowledge graph and a Streamlit demo. It is designed to be resume-ready, interview-defensible, and enterprise-presentable.

## Project Overview

The system ingests multi-format documents (PDF, DOCX, TXT, CSV), normalizes text with metadata, performs semantic-safe chunking, and indexes content with hybrid sparse (BM25) and dense (FAISS) retrieval. A lightweight knowledge graph captures entity co-occurrences to enrich answers and reasoning. An agentic controller orchestrates planning, retrieval, verification, and answer generation with citation tracing. A Streamlit UI demonstrates ingestion, retrieval, KG preview, and chat-style QA.

## Architecture

```
helperai-agentic-rag/
  app_streamlit.py          # Streamlit demo UI
  requirements.txt          # Dependencies
  src/
    ingest.py               # Document ingestion and normalization
    chunking.py             # Semantic-safe chunking with overlap
    bm25_index.py           # Sparse retrieval using BM25
    faiss_index.py          # Dense retrieval using FAISS and sentence embeddings
    hybrid_retriever.py     # Hybrid retrieval orchestration and scoring
    kg_builder.py           # Lightweight knowledge graph construction and traversal
    llm.py                  # LLM wrapper (Ollama + extractive fallback)
    agents.py               # Planner, Retriever, Verifier agents and controller
```

### How the Agentic RAG Works
1. **Ingestion** reads documents of multiple formats, normalizes text, and attaches metadata (document ID, source, page/row, chunk ID).
2. **Chunking** creates overlapping ~500-character chunks to preserve semantic continuity.
3. **Indexing** builds both BM25 (sparse) and FAISS (dense) indices for hybrid retrieval.
4. **Knowledge Graph** extracts simple entities per chunk, normalizes them, and builds co-occurrence edges to support graph-based evidence.
5. **Agentic Loop**: The Planner decomposes user queries, the Retriever gathers hybrid and KG evidence, and the Verifier checks grounding before answer generation.
6. **Answering** uses a local LLM via Ollama when available; otherwise, an extractive summarizer uses retrieved evidence. Answers include inline citations and an Evidence Trace.

### How the Knowledge Graph Improves Retrieval
- Entity-level nodes capture people, organizations, and capitalized noun phrases.
- Co-occurrence edges connect entities appearing together in chunks, enabling k-hop traversal.
- Query-time traversal pulls supporting facts aligned with extracted entities from the query, improving recall and grounding.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use CPU-only environments, the provided `faiss-cpu` and `sentence-transformers` work without CUDA.

## Running the Streamlit Demo

```bash
streamlit run app_streamlit.py
```

The UI lets you upload documents, observe ingestion status, preview the knowledge graph, and chat with the agentic QA system. Citations, retrieved chunks, and verification status are displayed for transparency.

## Configuration
- **LLM model**: set `OLLAMA_MODEL` environment variable (defaults to `llama3`).
- **Temperature**: set `LLM_TEMPERATURE` environment variable for deterministic control (default `0.0`).
- **Retrieval sizes**: tune `BM25_TOP_K`, `FAISS_TOP_K`, and `HYBRID_TOP_K` in `agents.py`.

## Interview-Ready Explanation
- Demonstrates multi-format ingestion, hybrid retrieval, and KG-augmented reasoning.
- Shows an agentic loop with planning, retrieval, verification, and answer generation.
- Includes fallback logic when an LLM is not reachable, ensuring robustness.
- Provides evidence tracing and simple evaluation hooks (faithfulness and coverage scores) to reason about answer quality.

## Notes
- Ollama must be running locally for generative answers; otherwise, the system falls back to extractive summarization with citations.
- The knowledge graph preview in Streamlit is textual for simplicity but can be extended to visual charts.
