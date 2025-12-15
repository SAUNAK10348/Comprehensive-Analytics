"""HelperAI – beginner-friendly RAG + analytics Streamlit app."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from analytics import plot_keyword_frequency  # type: ignore  # noqa: E402
from document_loader import chunk_text, load_document  # type: ignore  # noqa: E402
from embedder import build_index  # type: ignore  # noqa: E402
from llm import ask_llm  # type: ignore  # noqa: E402
from retriever import retrieve  # type: ignore  # noqa: E402
from scraper import save_raw_text, scrape_web  # type: ignore  # noqa: E402

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = BASE_DIR / "embeddings" / "faiss_index"


for directory in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
    directory.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="HelperAI – Analytics Assistant", page_icon="🤖")
st.title("HelperAI – Analytics Assistant")
st.caption("Web + docs → embeddings → retrieval → LLM → reports + charts")


if "texts" not in st.session_state:
    st.session_state.texts: List[str] = []
if "chunks" not in st.session_state:
    st.session_state.chunks: List[str] = []
if "index" not in st.session_state:
    st.session_state.index = None


st.sidebar.header("1) Ingest data")
url = st.sidebar.text_input("Public URL to scrape")
if st.sidebar.button("Add URL") and url:
    with st.spinner("Scraping web page..."):
        text = scrape_web(url)
        raw_path = RAW_DIR / f"web_{len(st.session_state.texts)+1}.txt"
        save_raw_text(text, raw_path)
        st.session_state.texts.append(text)
        st.sidebar.success(f"Added text from {url}")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF, DOCX, or TXT)", type=["pdf", "docx", "doc", "txt"], accept_multiple_files=True
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        dest = RAW_DIR / uploaded_file.name
        dest.write_bytes(uploaded_file.getbuffer())
        text = load_document(dest)
        st.session_state.texts.append(text)
    st.sidebar.success(f"Loaded {len(uploaded_files)} document(s)")


if st.sidebar.button("Build knowledge base"):
    if not st.session_state.texts:
        st.sidebar.error("Add at least one source first.")
    else:
        with st.spinner("Chunking and embedding text..."):
            chunks = []
            for i, text in enumerate(st.session_state.texts, start=1):
                text_path = PROCESSED_DIR / f"document_{i}.txt"
                text_path.write_text(text, encoding="utf-8")
                chunks.extend(chunk_text(text))

            index, _ = build_index(chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.success(f"Indexed {len(chunks)} chunks. Ready for questions!")


st.header("2) Ask questions")
query = st.text_input("Ask a question about your ingested content")
run_query = st.button("Run")

if run_query:
    if not st.session_state.index:
        st.error("Please build the knowledge base first.")
    elif not query.strip():
        st.error("Enter a question to run retrieval.")
    else:
        with st.spinner("Retrieving and asking the LLM..."):
            docs = retrieve(query, st.session_state.index, st.session_state.chunks)
            context = "\n".join(docs)
            answer = ask_llm(query, context)

        st.subheader("Answer")
        st.write(answer)

        if docs:
            st.subheader("Insights")
            fig = plot_keyword_frequency(docs)
            st.pyplot(fig)
            st.caption("Top keywords among the retrieved chunks.")


st.sidebar.header("Quick tips")
st.sidebar.markdown(
    "- Add URLs or upload PDF/DOCX/TXT files, then click **Build knowledge base**.\n"
    "- Set the `OPENAI_API_KEY` environment variable before running: `export OPENAI_API_KEY=...`.\n"
    "- Use concise questions for the best retrieval and answers."
)
