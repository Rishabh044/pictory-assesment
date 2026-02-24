# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semantic Sentence Search System - A FastAPI-based HTTP API for semantic sentence search using LangChain, Chroma vector store, and OpenAI embeddings. The system ingests documents, splits them into sentences, vectorizes them, and provides semantic search capabilities.

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run tests
pytest tests/
pytest tests/ -v                    # verbose
pytest tests/ --cov=app             # with coverage
pytest tests/test_search.py -k "test_name"  # single test
```

## Architecture

```
API Layer (FastAPI routes: /api/v1/ingest, /api/v1/search)
    ↓
Service Layer (ingestion_service.py, search_service.py)
    ↓
Core Layer (sentence_splitter.py, vector_store.py, embeddings.py, document_processor.py)
    ↓
Data Layer (Chroma vector store + file system)
```

### Key Components

- **app/core/sentence_splitter.py**: Uses LangChain's `NLTKTextSplitter` for sentence boundary detection. Generates stable sentence IDs in format `{document_id}::{sentence_index}::{content_hash[:8]}`
- **app/core/vector_store.py**: Chroma wrapper with `add_documents()`, `similarity_search_with_score()`, and `get_similar_to_document()` methods
- **app/services/search_service.py**: Implements dual-search logic - (1) top K matches via vector similarity, (2) sentences similar to top match (excluding self)

### Sentence Metadata Structure
```python
{
    "document_id": "doc_name.txt",
    "sentence_id": "doc_name.txt::12::f3a2b1c9",
    "sentence_text": "Text of sentence",
    "sentence_index": 12,
    "start_offset": 450,  # character position
    "end_offset": 520
}
```

## LangChain Integration

```python
# Text splitting
from langchain.text_splitter import NLTKTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector store
from langchain_community.vectorstores import Chroma

# Document loaders
from langchain_community.document_loaders import TextLoader, PyPDFLoader
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/ingest` | POST | Ingest documents from folder, split into sentences, index |
| `/api/v1/search` | POST | Semantic search with dual-search logic |
| `/health` | GET | Health check |

## Configuration

Uses Pydantic Settings via `app/config.py`. Required env vars:
- `OPENAI_API_KEY`

Optional: `CHROMA_PERSIST_DIRECTORY` (default: `./data/chroma_db`)
