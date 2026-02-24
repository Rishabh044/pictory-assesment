# Semantic Sentence Search API

A FastAPI service that ingests PDF documents, splits them into sentences, indexes them with OpenAI embeddings in a Chroma vector store, and exposes a semantic search endpoint.

## Prerequisites

- Docker & Docker Compose **or** Python 3.11+
- An OpenAI API key

---

## Running with Docker (recommended)

### 1. Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

Or create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

### 2. Build and start

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`.

### 3. Stop

```bash
docker compose down
```

Chroma data persists in `./data/chroma_db` between restarts.

---

## Running locally (without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload
```

---

## Running tests

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## API Endpoints

### Health check

```bash
curl http://localhost:8000/health
```

### Ingest a PDF

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/documents/sample.pdf"}'
```

The file path must be accessible inside the container. Place PDFs in `./data/documents/` — it is mounted into the container automatically.

### Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search text here", "k": 5}'
```

`k` is optional (default 5, range 1–50).

---

## Configuration

| Environment variable       | Default            | Description                        |
|----------------------------|--------------------|------------------------------------|
| `OPENAI_API_KEY`           | *(required)*       | OpenAI API key                     |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma_db` | Where Chroma stores its index      |
| `EMBEDDING_MODEL`          | `text-embedding-3-small` | OpenAI embedding model        |

---

## Project structure

```
app/
  api/
    models/        # Pydantic request/response schemas
    routes/        # FastAPI route handlers (ingest, search)
  core/            # document_processor, sentence_splitter, vector_store, embeddings
  errors/          # Custom error classes
  services/        # ingestion_service, search_service
  config.py        # Settings (loaded from env)
  main.py          # FastAPI app entry point
tests/             # pytest test suite
```
