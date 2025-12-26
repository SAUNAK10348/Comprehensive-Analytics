# HelperAI Agentic RAG + Ontology-Driven Knowledge Graph

An enterprise-grade, end-to-end Retrieval-Augmented Generation (RAG) system with ontology alignment, hybrid retrieval (BM25 + FAISS), ontology-aware Knowledge Graph traversal, agentic planning/verification, and a Streamlit UI. Built for deterministic, resume-ready demos and interview defense.

## Project Overview
This project ingests multi-format documents (PDF, DOCX, TXT, CSV), aligns content to a controlled ontology, builds a knowledge graph with validation, indexes text for hybrid retrieval, and orchestrates a planner → retriever → verifier agent loop to produce citation-grounded answers. It includes observability hooks (latency, faithfulness), ontology lenses, and safeguards for prompt injection. A Hugging Face login helper makes it easy to pull local models in restricted environments.

## Architecture (ASCII)
```
                       +---------------------------+
                       |        Streamlit UI       |
                       | Upload | Config | Chat    |
                       +---------------------------+
                                      |
                                      v
+---------------------------+   +---------------+   +--------------------+
|       Ingestion           |-->|   Chunking    |-->|   Hybrid Indexes   |
| PDF/DOCX/TXT/CSV + Cache  |   | 800-1200 chars|   | BM25 + FAISS       |
+---------------------------+   +---------------+   +--------------------+
           |                               |                   |
           v                               v                   v
+---------------------------+        +--------------------------------------+
|  Ontology Alignment       |        |  Knowledge Graph (NetworkX)          |
| Normalize entities/edges  |<-------| Validated nodes/edges + provenance   |
+---------------------------+        +--------------------------------------+
                                      |
                                      v
                           +-----------------------------+
                           | Agentic Controller          |
                           | Planner → Retriever → LLM   |
                           | Verifier (faithfulness)     |
                           +-----------------------------+
                                      |
                                      v
                           +-----------------------------+
                           |  Answer + Citations + KG    |
                           |  Evidence Trace + Verdict   |
                           +-----------------------------+
```

## How Ontology-Driven RAG Works
- **Controlled vocabulary:** `ontology/enterprise_ontology.json` defines entity types, relation types, allowed domains/ranges, and synonyms to normalize noisy inputs.
- **Ontology-aware KG:** Entities and relations are extracted with heuristics, validated against constraints, scored for confidence, and stored with provenance.
- **Intent-guided retrieval:** Planner tags queries with ontology intents (e.g., Policy, Metric) to bias both text retrieval and KG traversal, increasing precision.
- **Verification:** The verifier checks every claim for citations and rejects prompts inside retrieved text (prompt-injection guard), forcing re-retrieval when unsupported.

## Hybrid Retrieval
1. **Sparse:** BM25 over chunked text.
2. **Dense:** Sentence-transformer embeddings + FAISS (with TF-IDF fallback if embeddings unavailable).
3. **Fusion:** Merge, de-duplicate by chunk_id, and re-rank with configurable weights from `config.py`.

## Knowledge Graph Traversal
- Nodes: ontology entity type, canonical name, attributes, provenance.
- Edges: validated relation type, confidence, citing chunks.
- Traversal: k-hop exploration constrained by ontology relations; returns graph evidence bound to source chunks for citations.

## Verifier & Hallucination Defense
- Checks that each sentence has citations.
- Compares cited chunk IDs to evidence and flags missing support.
- Strips instruction-like text from context to avoid prompt injection.
- Provides a PASS/FAIL verdict with reasons plus coverage metrics.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r helperai-agentic-rag/requirements.txt
```

## Running the App
```bash
cd helperai-agentic-rag
export OLLAMA_MODEL=llama3       # or another local Ollama model
streamlit run app_streamlit.py
```

### Hugging Face Login & Local Models
The LLM module exposes `ensure_hf_login()` to authenticate (token via `HUGGINGFACE_TOKEN`) and download models locally when internet or gated access is required.
For Codespaces CPU/GPU smoke tests without Kaggle, you can preload a Llama pipeline:
```bash
cd helperai-agentic-rag
export HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
python -m src.load_llama_pipeline --model meta-llama/Llama-3-8b-instruct --prompt "hello codespaces"
```

## Configuration
- `src/config.py` centralizes tunables (chunk sizes, retrieval weights, KG hops, temperatures, file size limits). All values can be overridden by environment variables.

## Observability & Evaluation
- Latency timers per stage (ingest, index, retrieve, LLM, verify).
- Faithfulness score (sentences with citations) and coverage (unique sources).
- Structured JSON logging with request IDs; basic redaction for emails/phones.

## Interview Defense (Talking Points)
- **Why ontology?** Normalizes noisy enterprise data, enforces relation constraints, reduces drift/hallucination by constraining KG and retrieval.
- **Trade-offs:** Rule-based extraction is lightweight but may miss nuance; FAISS on CPU is fast for demos but might need GPU for scale; Ollama fallback keeps offline resilience.
- **Failure modes:** Poor OCR in PDFs; unnormalized synonyms; thin evidence causing verifier fails; embeddings fallback (TF-IDF) less semantic.
- **Next steps:** Swap KG to Neo4j with RAGAS metrics; add active learning for entity normalization; plug in GPU embeddings; enhance prompt-injection detection.

## Repository Layout
```
helperai-agentic-rag/
  app_streamlit.py
  requirements.txt
  README.md
  instructions.txt
  ontology/
    enterprise_ontology.json
  src/
    config.py
    ingest.py
    chunking.py
    bm25_index.py
    faiss_index.py
    hybrid_retriever.py
    ontology.py
    kg_builder.py
    agents.py
    llm.py
    eval_utils.py
```

## Running Tests / Checks
This project is designed for interactive use; no formal test suite is shipped. Use `streamlit run app_streamlit.py` to exercise ingestion, retrieval, KG, and verification end-to-end.
