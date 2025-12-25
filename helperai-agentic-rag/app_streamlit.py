"""Streamlit UI for the Agentic RAG + Knowledge Graph demo."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.agents import AgentOrchestrator
from src.hybrid_retriever import HybridRetriever
from src.ingest import DocumentChunk, DocumentIngestor
from src.kg_builder import KnowledgeGraphBuilder
from src.llm import LLMClient


st.set_page_config(page_title="HelperAI Agentic RAG", layout="wide")
st.title("HelperAI Agentic RAG + KG Demo")

if "chunks" not in st.session_state:
    st.session_state.chunks: List[DocumentChunk] = []
if "kg" not in st.session_state:
    st.session_state.kg = KnowledgeGraphBuilder()
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever()
if "built" not in st.session_state:
    st.session_state.built = False
if "agent" not in st.session_state:
    st.session_state.agent = None


def ingest_files(uploaded_files) -> None:
    ingestor = DocumentIngestor()
    for file in uploaded_files:
        data = file.read()
        path = Path(file.name)
        temp_path = Path("/tmp") / path.name
        temp_path.write_bytes(data)
        chunks = ingestor.ingest(temp_path)
        st.session_state.chunks.extend(chunks)
        st.success(f"Ingested {file.name} into {len(chunks)} chunks")
    st.session_state.kg.add_documents(st.session_state.chunks)
    st.session_state.retriever.build(st.session_state.chunks)
    llm = LLMClient()
    st.session_state.agent = AgentOrchestrator(
        retriever=st.session_state.retriever,
        kg=st.session_state.kg,
        llm=llm,
    )
    st.session_state.built = True


with st.sidebar:
    st.header("Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or CSV documents",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )
    if st.button("Ingest Documents"):
        if uploaded_files:
            ingest_files(uploaded_files)
        else:
            st.warning("Please upload at least one document.")
    st.markdown("---")
    st.subheader("Knowledge Graph Preview")
    if st.session_state.kg.graph.number_of_nodes() > 0:
        preview = st.session_state.kg.summary(limit=10)
        for node, deg in preview:
            st.write(f"{node} (degree {deg})")
    else:
        st.write("No graph yet. Ingest documents first.")

st.markdown("## Chat")
query = st.text_input("Ask a question about your documents")
if st.button("Run Agent"):
    if not st.session_state.built:
        st.error("Please ingest documents first.")
    elif not query.strip():
        st.warning("Enter a question.")
    else:
        result = st.session_state.agent.answer(query)
        st.markdown("### Answer")
        st.write(result["answer"])
        st.markdown(f"**Verified:** {'✅' if result['verified'] else '⚠️'}")
        st.markdown("### Retrieved Chunks")
        for chunk in result["evidence"]:
            meta = chunk.metadata or {}
            st.code(
                f"[{chunk.chunk_id}] Source: {chunk.source}, Page: {chunk.page}, Row: {chunk.row}\n{chunk.text}",
                language="markdown",
            )
        st.markdown("### Evidence Trace")
        trace = result["trace"]
        st.json(trace)
