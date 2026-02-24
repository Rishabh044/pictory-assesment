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

---

## Sample Usage

### Example 1: Ingest and Search

**Step 1: Build the Index**

Place a PDF file in `./data/documents/research.pdf`, then ingest it:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/documents/research.pdf"}'
```

**Response:**
```json
{
  "document_id": "research_a3f8d2e1",
  "sentences_count": 42,
  "status": "success"
}
```

**Step 2: Run a Query**

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main benefits of renewable energy?", "k": 3}'
```

**Response:**
```json
{
  "query": "What are the main benefits of renewable energy?",
  "top_matches": [
    {
      "sentence_id": "research_a3f8d2e1::15::c4f7b2a9",
      "sentence_text": "Renewable energy sources reduce greenhouse gas emissions and provide sustainable alternatives to fossil fuels.",
      "document_id": "research_a3f8d2e1",
      "sentence_index": 15,
      "score": 0.82
    },
    {
      "sentence_id": "research_a3f8d2e1::23::d9e1c3f2",
      "sentence_text": "Solar and wind power offer long-term economic benefits through reduced operating costs.",
      "document_id": "research_a3f8d2e1",
      "sentence_index": 23,
      "score": 0.76
    },
    {
      "sentence_id": "research_a3f8d2e1::8::b2a4e6c1",
      "sentence_text": "Clean energy technologies have become increasingly cost-competitive in recent years.",
      "document_id": "research_a3f8d2e1",
      "sentence_index": 8,
      "score": 0.71
    }
  ],
  "time_elapsed_seconds": 0.1243
}
```

---

## Design Documentation

### Architecture Overview

The system follows a **pipeline architecture** with clear separation of concerns:

```
┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 1. Document Processor (DocumentProcessor)          │
│    - Validates PDF existence and format             │
│    - Extracts text using PyPDFLoader                │
│    - Generates content-based document ID            │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 2. Sentence Splitter (SentenceSplitter)            │
│    - Tokenizes text into sentences using NLTK       │
│    - Creates stable IDs with content hashes         │
│    - Tracks offsets and metadata                    │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 3. Embeddings Service (EmbeddingsService)          │
│    - Generates 1536-dim vectors via OpenAI API      │
│    - Uses text-embedding-3-small model              │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 4. Vector Store (Chroma + VectorStoreManager)      │
│    - Stores embeddings with HNSW indexing           │
│    - Persists to disk in ./data/chroma_db           │
│    - Enables fast approximate nearest neighbor      │
└─────────────────────────────────────────────────────┘
```

**Query Flow:**
```
User Query → Embeddings Service → Vector Store → Similarity Search → Ranked Results
```

The system uses **async/await** throughout for non-blocking I/O operations, leveraging `asyncio.to_thread()` for CPU-bound tasks like PDF parsing and sentence splitting.

### Sentence Segmentation Approach

**Technology:** NLTK's Punkt sentence tokenizer (`nltk.sent_tokenize`)

**Why NLTK Punkt?**
- Pre-trained ML model for sentence boundary detection
- Handles edge cases: abbreviations (Dr., etc.), decimals, quotes
- Language-aware (defaults to English)
- Lightweight and reliable for general text

**Metadata Tracking:**
Each sentence includes:
- `sentence_id`: Stable identifier in format `{doc_id}::{index}::{hash[:8]}`
- `sentence_index`: Position in original document
- `start_offset` / `end_offset`: Character positions for text highlighting
- `document_id`: Source document reference

**Alternative Considered:**
- spaCy's sentence segmentation: More accurate but heavier dependency, requires model download
- Regex-based splitting: Fragile for complex punctuation patterns

### Embedding + Indexing Approach

**Embedding Model:** OpenAI `text-embedding-3-small`
- **Dimensionality:** 1536
- **Max tokens:** 8192 per input
- **Advantages:** High quality, fast inference, cost-effective ($0.02/1M tokens)
- **Context window:** Sufficient for most sentences without truncation

**Vector Store:** ChromaDB with HNSW (Hierarchical Navigable Small World)
- **Index type:** HNSW graph for approximate nearest neighbor (ANN) search
- **Distance metric:** Cosine similarity (see below)
- **Persistence:** Local disk storage enables stateful operation across restarts
- **Collection:** Single collection `sentence_embeddings` for all documents

**Indexing Pipeline:**
1. Each sentence → OpenAI API → 1536-dimensional vector
2. Batch insert into Chroma collection (LangChain handles batching)
3. HNSW index automatically updated for efficient retrieval

**Alternative Considered:**
- FAISS: More control but requires manual index management
- Pinecone/Weaviate: Cloud-hosted but adds external dependency

### Similarity Metric Choice

**Metric:** **Cosine Similarity**

**Justification:**
- **Semantic focus:** Cosine measures angle between vectors, ignoring magnitude
  - Two sentences with similar meaning but different lengths get high similarity
  - Ideal for text embeddings where direction encodes semantics
- **OpenAI recommendation:** text-embedding-3-small is optimized for cosine similarity
- **Normalization:** OpenAI embeddings are normalized, making cosine equivalent to dot product (but more interpretable)

**Score Interpretation:**
- `1.0`: Identical meaning
- `0.8-0.9`: Very similar
- `0.6-0.8`: Moderately related
- `<0.6`: Weakly related

**Alternative Considered:**
- Euclidean distance: Less effective for high-dimensional semantic spaces (suffers from curse of dimensionality)

### Limitations and Future Improvements

**Current Limitations:**

1. **Language Support**
   - Only English sentence segmentation (NLTK Punkt default)
   - OpenAI embeddings work multi-lingually but not optimized

2. **Scalability**
   - In-memory Chroma not ideal for millions of sentences
   - No horizontal scaling (single instance)
   - Synchronous embedding calls (no batching optimization)

3. **PDF Handling**
   - No OCR support for scanned PDFs
   - Tables/figures ignored
   - Multi-column layouts may scramble sentence order

4. **Search Features**
   - No filtering by document, date, or metadata
   - No hybrid search (semantic + keyword)
   - No result re-ranking or relevance feedback

5. **Observability**
   - No metrics collection (query latency, embedding costs)
   - No distributed tracing

**Future Improvements:**

1. **Multi-document Management**
   - Add document deletion/update endpoints
   - Implement metadata filtering in search
   - Support incremental updates without re-embedding

2. **Performance Optimization**
   - Batch OpenAI API calls (up to 2048 inputs per request)
   - Add caching layer for repeated queries
   - Implement connection pooling for Chroma

3. **Advanced Search**
   - Hybrid search combining BM25 (keyword) + semantic
   - MMR (Maximal Marginal Relevance) for diverse results
   - Context expansion (return surrounding sentences)

4. **Production Readiness**
   - Add authentication/authorization
   - Implement rate limiting
   - Deploy Chroma in client-server mode for horizontal scaling
   - Add monitoring with Prometheus/Grafana
   - Implement retry logic with exponential backoff

5. **Enhanced PDF Support**
   - OCR integration (Tesseract, Azure Form Recognizer)
   - Table extraction (Camelot, Tabula)
   - Preserve document structure metadata

6. **Alternative Embeddings**
   - Support local models (Sentence-BERT) for privacy/cost
   - Experiment with larger OpenAI models (text-embedding-3-large) for critical use cases
   - Fine-tune embeddings on domain-specific data

---

## License

MIT
