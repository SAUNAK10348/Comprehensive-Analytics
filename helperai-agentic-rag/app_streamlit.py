"""
Streamlit app for ontology-driven Agentic RAG.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from src.agents import AgentOrchestrator
from src.config import CONFIG
from src.hybrid_retriever import HybridRetriever
from src.ingest import ingest_document
from src.kg_builder import KnowledgeGraphBuilder
from src.ontology import Ontology


st.set_page_config(page_title="HelperAI Agentic RAG", layout="wide")


def _write_upload(file) -> str:
    upload_dir = Path(CONFIG.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / file.name
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return str(path)


def initialize_state() -> None:
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None
    if "kg_builder" not in st.session_state:
        st.session_state["kg_builder"] = None
    if "ontology" not in st.session_state:
        st.session_state["ontology"] = Ontology()


def sidebar_config() -> None:
    st.sidebar.header("Configuration")
    CONFIG.top_k_sparse = st.sidebar.slider("Top-K Sparse", 1, 20, CONFIG.top_k_sparse)
    CONFIG.top_k_dense = st.sidebar.slider("Top-K Dense", 1, 20, CONFIG.top_k_dense)
    CONFIG.weight_sparse = st.sidebar.slider("Weight Sparse", 0.0, 1.0, float(CONFIG.weight_sparse))
    CONFIG.weight_dense = st.sidebar.slider("Weight Dense", 0.0, 1.0, float(CONFIG.weight_dense))
    CONFIG.kg_max_hops = st.sidebar.slider("KG Max Hops", 1, 3, CONFIG.kg_max_hops)
    CONFIG.kg_max_neighbors = st.sidebar.slider("KG Max Neighbors", 3, 20, CONFIG.kg_max_neighbors)
    CONFIG.ollama_model = st.sidebar.text_input("Ollama Model", CONFIG.ollama_model)
    CONFIG.temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, float(CONFIG.temperature))
    CONFIG.max_tokens = st.sidebar.slider("LLM Max Tokens", 128, 1024, CONFIG.max_tokens, step=64)


def build_indexes(file_paths: List[str]) -> None:
    with st.spinner("Ingesting and building indexes..."):
        chunks = []
        ingest_latency = 0.0
        for path in file_paths:
            t0 = time.time()
            chunks.extend(ingest_document(path))
            ingest_latency += time.time() - t0
        retriever = HybridRetriever()
        retriever.build(chunks)
        kg_builder = KnowledgeGraphBuilder(st.session_state["ontology"])
        kg_builder.build_from_chunks(chunks)
        st.session_state["chunks"] = chunks
        st.session_state["retriever"] = retriever
        st.session_state["kg_builder"] = kg_builder
        st.success(f"Ingested {len(chunks)} chunks • KG nodes: {kg_builder.graph.number_of_nodes()} • edges: {kg_builder.graph.number_of_edges()} • ingest latency: {ingest_latency:.2f}s")


def main() -> None:
    initialize_state()
    st.title("HelperAI Ontology-Driven Agentic RAG")
    sidebar_config()

    st.markdown("Upload files (PDF/DOCX/TXT/CSV), then build the index and ask questions.")
    uploaded = st.file_uploader("Upload documents", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True)
    uploaded_paths: List[str] = []
    if uploaded:
        for file in uploaded:
            uploaded_paths.append(_write_upload(file))

    if st.button("Build index and KG", use_container_width=True) and uploaded_paths:
        build_indexes(uploaded_paths)

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Ask a question", placeholder="e.g., Which systems depend on the Payments API and what policies govern them?")
        if st.button("Run", use_container_width=True) and query:
            if not st.session_state["retriever"]:
                st.error("Please upload and build the index first.")
            else:
                orchestrator = AgentOrchestrator(
                    retriever=st.session_state["retriever"],
                    kg_builder=st.session_state["kg_builder"],
                    ontology=st.session_state["ontology"],
                )
                with st.spinner("Running agentic loop..."):
                    result = orchestrator.run(query)
                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Verification")
                st.json(result.get("verification", {}))

                st.subheader("Citations & Evidence Chunks")
                for chunk in result["evidence_chunks"][:15]:
                    st.markdown(f"**{chunk.get('chunk_id')}** ({chunk.get('source')})")
                    st.write(chunk.get("text"))

                st.subheader("KG Evidence")
                st.json(result.get("kg_evidence", []))

                st.subheader("Ontology Lens")
                st.write(result.get("plan", {}).get("ontology_lens", []))

                st.subheader("Timings")
                st.json(result.get("timings", {}))

    with col2:
        st.markdown("### Knowledge Graph Preview")
        if st.session_state["kg_builder"]:
            summary = st.session_state["kg_builder"].summary()
            st.metric("Nodes", summary["nodes"])
            st.metric("Edges", summary["edges"])
        if st.session_state["chunks"]:
            df = pd.DataFrame(st.session_state["chunks"])[:50]
            st.dataframe(df[["chunk_id", "source", "page", "row"]])


if __name__ == "__main__":
    main()
