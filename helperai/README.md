# HelperAI – Beginner RAG + Analytics

A compact Retrieval-Augmented Generation (RAG) starter kit that ingests web pages and PDF/DOCX/TXT files, builds embeddings with FAISS, answers questions with an LLM, and shows quick keyword charts via Streamlit.

## Project structure
```
helperai/
├── data/
│   ├── raw/            # scraped pages, uploaded docs
│   ├── processed/      # cleaned text
├── embeddings/
│   └── faiss_index/
├── src/
│   ├── scraper.py
│   ├── document_loader.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── llm.py
│   ├── analytics.py
├── app.py              # Streamlit UI
└── requirements.txt
```

## Setup (one go)
1. Create and activate a virtual environment if desired.
2. Install dependencies and run Streamlit:
   ```bash
   pip install -r helperai/requirements.txt
   export OPENAI_API_KEY=your_key_here
   streamlit run helperai/app.py
   ```

## Using the app
1. In the left sidebar, add one or more sources:
   - Paste a public URL to scrape article text.
   - Upload PDF, DOCX, or TXT files (TXT is fully supported out of the box).
2. Click **Build knowledge base** to chunk the text and create embeddings locally with FAISS.
3. Ask a question in the main panel and click **Run**. HelperAI retrieves relevant chunks, asks the LLM, and plots the top keywords.

## Under the hood
- **Scraping:** `requests` + `BeautifulSoup` (`src/scraper.py`).
- **Docs:** `pdfplumber`, `python-docx`, and TXT reader (`src/document_loader.py`).
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS (`src/embedder.py`).
- **Retrieval:** semantic search against the FAISS index (`src/retriever.py`).
- **LLM:** OpenAI chat completion with contextual prompt (`src/llm.py`).
- **Analytics:** quick keyword frequency bar chart (`src/analytics.py`).

## Notes
- The app stores ingested raw and processed text under `helperai/data/` so you can inspect or reuse it.
- To reset the session, simply reload the page; to start over, clear the folders under `data/`.
- You can extend this base to add CSV ingestion, PDF/DOCX report export, or a persistent FAISS index loader.
