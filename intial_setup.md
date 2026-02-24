# Implementation Plan: Semantic Sentence Search System

## Overview
Build a FastAPI-based HTTP API for semantic sentence search using LangChain, Chroma vector store, and OpenAI embeddings. The system will ingest documents, split them into sentences, vectorize them, and provide semantic search capabilities.

## Tech Stack
- **Framework**: Python + FastAPI
- **Vector Store**: Chroma (via LangChain)
- **Embeddings**: OpenAI Embeddings
- **Orchestration**: LangChain for all vector operations

## Project Structure

```
pictory-assesment/
├── CLAUDE.md                          # Project overview and architecture
├── IMPL_PLAN.md                       # Detailed implementation guide
├── README.md                          # Setup and usage instructions
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore file
│
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI application entry
│   ├── config.py                      # Pydantic Settings configuration
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py              # POST /api/v1/ingest endpoint
│   │   │   └── search.py              # POST /api/v1/search endpoint
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── request.py             # Request schemas
│   │       └── response.py            # Response schemas
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_processor.py     # LangChain document loaders
│   │   ├── sentence_splitter.py      # Sentence-level text splitting
│   │   ├── vector_store.py           # Chroma vector store manager
│   │   └── embeddings.py             # OpenAI embeddings config
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion_service.py      # Ingestion orchestration
│   │   └── search_service.py         # Search orchestration
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py                 # Logging configuration
│
├── data/
│   ├── documents/                     # Input documents
│   └── chroma_db/                     # Chroma persistence
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_ingestion.py
    ├── test_search.py
    └── test_sentence_splitter.py
```

## Critical Files (Implementation Order)

### 1. Configuration & Setup Files

**File: `requirements.txt`**
- Dependencies: fastapi, uvicorn, langchain, langchain-community, langchain-openai, chromadb, openai, pypdf, pydantic, pydantic-settings, python-dotenv
- Testing: pytest, pytest-asyncio, httpx

**File: `.env.example`**
- OpenAI API key template
- Configuration defaults

**File: `.gitignore`**
- Exclude: data/, .env, __pycache__, chroma_db/

**File: `app/config.py`**
- Use Pydantic Settings for configuration
- Environment variables: OPENAI_API_KEY, CHROMA_PERSIST_DIRECTORY, etc.

### 2. Core Components (LangChain Integration)

**File: `app/core/sentence_splitter.py`** ⭐ CRITICAL
- Use `NLTKTextSplitter` from LangChain for sentence boundary detection
- Generate stable sentence IDs: `{document_id}::{index}::{hash}`
- Return LangChain Document objects with metadata:
  - document_id, sentence_id, sentence_text, sentence_index
  - start_offset, end_offset (character positions)
- Hash-based ID ensures stability across re-ingestion

**File: `app/core/embeddings.py`**
- Configure `OpenAIEmbeddings` from langchain-openai
- Model: "text-embedding-3-small"
- Load API key from config

**File: `app/core/vector_store.py`** ⭐ CRITICAL
- Initialize `Chroma` from langchain-community.vectorstores
- Methods:
  - `add_documents(documents: List[Document])` - batch insert sentences
  - `similarity_search_with_score(query, k, filter)` - vector search with scores
  - `get_similar_to_document(document, k)` - find similar sentences
- Persist to disk after each ingestion

**File: `app/core/document_processor.py`**
- Use `TextLoader` for .txt files
- Use `PyPDFLoader` for .pdf files (optional)
- Use `DirectoryLoader` for batch loading
- Return dict mapping document_id → content

### 3. Service Layer (Orchestration)

**File: `app/services/ingestion_service.py`** ⭐ CRITICAL
- Pipeline:
  1. Load documents via DocumentProcessor
  2. Split into sentences via SentenceSplitter
  3. Add to vector store with embeddings
- Handle errors gracefully (log failures, continue with others)

**File: `app/services/search_service.py`** ⭐ CRITICAL
- Implements dual-search logic:
  1. **Top K matches**: `vector_store.similarity_search(query, k=top_k)`
  2. **Similar to top match**: Use top result's text as query to find N similar sentences
- Exclude self from similar sentences
- Return structured response with scores

### 4. API Layer (FastAPI)

**File: `app/api/models/request.py`**
- `IngestRequest`: folder_path, file_patterns, regenerate
- `SearchRequest`: query, top_k (default 5), similar_n (default 5), filter

**File: `app/api/models/response.py`**
- `SentenceMatch`: document_id, sentence_id, sentence_text, score, metadata
- `IngestResponse`: status, documents_processed, sentences_indexed, time_elapsed_seconds, details
- `SearchResponse`: query, top_matches, similar_to_top_match

**File: `app/api/routes/ingest.py`**
- POST /api/v1/ingest
- Validate folder path exists
- Call IngestionService
- Return detailed statistics

**File: `app/api/routes/search.py`**
- POST /api/v1/search
- Validate query not empty
- Call SearchService
- Return JSON with top_matches and similar_to_top_match

**File: `app/main.py`** ⭐ CRITICAL
- Create FastAPI app
- Add CORS middleware
- Include routers
- Health check endpoint
- Root endpoint with API documentation

### 5. Documentation Files

**File: `CLAUDE.md`**
- Project overview and architecture
- LangChain component usage
- API endpoints specification
- Design decisions (sentence splitting, ID generation, similarity search)
- How to extend the system

**File: `IMPL_PLAN.md`**
- Step-by-step implementation guide
- Code examples for each component
- LangChain integration details
- Testing strategy
- Deployment considerations

**File: `README.md`**
- Quick start guide
- Installation instructions
- Environment setup
- API usage examples (curl/Python)
- Troubleshooting

## Implementation Steps

### Phase 1: Project Setup (30 min)
1. Create directory structure
2. Create virtual environment: `python -m venv venv`
3. Create `requirements.txt` with all dependencies
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env.example` and `.gitignore`
6. Create empty `__init__.py` files

### Phase 2: Configuration (15 min)
1. Implement `app/config.py` using Pydantic Settings
2. Create `.env` file with OPENAI_API_KEY
3. Implement `app/utils/logger.py` for structured logging

### Phase 3: Core Components (90 min)
1. Implement `app/core/embeddings.py`
   - Configure OpenAIEmbeddings
2. Implement `app/core/sentence_splitter.py`
   - NLTKTextSplitter integration
   - Sentence ID generation logic
   - Metadata extraction (offsets)
3. Implement `app/core/vector_store.py`
   - Chroma initialization with persistence
   - Add documents method
   - Similarity search methods
4. Implement `app/core/document_processor.py`
   - TextLoader for .txt
   - PyPDFLoader for .pdf
   - Directory scanning

### Phase 4: Service Layer (60 min)
1. Implement `app/services/ingestion_service.py`
   - Document loading → splitting → embedding → storage pipeline
   - Error handling and logging
   - Statistics collection
2. Implement `app/services/search_service.py`
   - Query-based search (top K)
   - Similarity-based search (similar to top match)
   - Result formatting

### Phase 5: API Layer (60 min)
1. Create Pydantic models in `app/api/models/`
2. Implement `app/api/routes/ingest.py`
3. Implement `app/api/routes/search.py`
4. Create `app/main.py` with FastAPI app
5. Test endpoints with sample data

### Phase 6: Documentation (45 min)
1. Write `CLAUDE.md` - architecture and design decisions
2. Write `IMPL_PLAN.md` - implementation guide with code examples
3. Write `README.md` - setup and usage instructions

### Phase 7: Testing (60 min)
1. Create test fixtures in `tests/conftest.py`
2. Write unit tests for sentence splitter
3. Write integration tests for API endpoints
4. Test with sample documents
5. Verify sentence ID stability

## Key LangChain Components

### mext Splitting
```python
from langchain.text_splitter import NLTKTextSplitter

splitter = NLTKTextSplitter()
sentences = splitter.split_text(text)
```

### Embeddings
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.openai_api_key
)
```

### Vector Store
```python
from langchain_community.vectorstores import Chroma

vector_store = Chroma(
    collection_name="sentence_embeddings",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db"
)

# Add documents
vector_store.add_documents(documents)

# Search
results = vector_store.similarity_search_with_score(query, k=5)
```

### Document Loading
```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader

loader = TextLoader("document.txt")
docs = loader.load()
```

## API Endpoints

### POST /api/v1/ingest
**Request:**
```json
{
  "file_path": "./data/documents/sample_data.pdf",
  "regenerate": true
}
```

**Response:**
on success: 
```json
{
  "status": "success",
  "sentences_indexed": 450,
  "time_elapsed_seconds": 12.5,
  "details": [
    {
      "document_id": "file_name + sha256(content)[:8]",
      "sentences_count": 45,
      "status": "success"
    }
  ]
}
``` 
on failure:
```json
{
  "status": "failure",
  "sentences_indexed": 0,
  "time_elapsed_seconds":0,
  "details": "error_details",
```
### POST /api/v1/search

{
  "query": "What is machine learning?",
  "top_k": 5,
}
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "top_matches": [
    {
      "document_id": "ml_intro.txt",
      "sentence_id": "ml_intro.txt::12::f3a2b1c9",
      "sentence_text": "Machine learning is a subset of AI...",
      "score": 0.92,
      "metadata": {"sentence_index": 12}
    }
  ]
}
```

## Design Decisions

### Sentence ID Stability
- Format: `{document_id}::{sentence_index}`
- Ensures same position = same ID across re-ingestion
- Traceable to source document
- Hash component handles minor text edits

### Metadata Storage
- Store comprehensive metadata in Chroma for:
  - Filtering by document
  - Tracing results back to source
  - Future enhancements (paragraph context, etc.)

### Error Handling
- different directory for error, with all the base errors in one file and the custom error in the file of thier own
- Custom Error messages for different services
- Log all errors with context
- Continue processing on document-level failures
- Return partial results when possible
- Validate inputs at API boundary - the inputs are strictly pdfs for now for `v1/ingest`

## Testing Strategy

### Unit Tests
- Sentence splitting with various formats
- Sentence ID generation stability
- Metadata extraction accuracy

### Integration Tests
- End-to-end ingestion workflow
- Search with known queries
- API response schemas

### Test Data
- Sample .pdf file 
- Known query-result pairs for validation

## Production Considerations

### Performance
- Use async/await for I/O operations
- Monitor OpenAI API rate limits
- Enable Chroma persistence

### Security
- Never commit .env file

## Success Criteria

✅ System ingests documents and splits into sentences
✅ Sentence IDs are stable across re-ingestion
✅ Search returns top K relevant sentences with scores
✅ JSON response format matches specification
✅ All metadata fields are populated correctly
✅ API is documented and easy to use
✅ Tests cover core functionality
✅ CLAUDE.md and IMPL_PLAN.md are comprehensive

## Estimated Timeline

- **Setup**: 30 minutes
- **Core Implementation**: 3-4 hours
- **API Layer**: 1 hour
- **Documentation**: 45 minutes
- **Testing**: 1 hour
- **Total**: ~6-7 hours for complete implementation

## Next Steps After Plan Approval

1. Create virtual environment and install dependencies
2. Implement configuration and setup files
3. Build core components (sentence splitter, vector store)
4. Implement service layer
5. Create FastAPI endpoints
6. Write documentation (CLAUDE.md, IMPL_PLAN.md)
7. Test end-to-end workflow
8. Create sample documents for demonstration
