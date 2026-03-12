# Interview Questions — Semantic Sentence Search API

## 1. Architecture & System Design

**Q1.1:** Describe the end-to-end data flow when a user ingests a PDF document. What are the key processing stages, and which module is responsible for each?

**Q1.2:** The project separates code into `api/`, `core/`, `services/`, and `errors/` layers. What is the responsibility of each layer, and why is this separation beneficial?

**Q1.3:** Why does the project use a service layer (`IngestionService`, `SearchService`) instead of putting all logic directly in the route handlers?

**Q1.4:** If you needed to support a second document type (e.g., DOCX), which modules would you modify and which would remain unchanged? What does this tell you about the architecture?

**Q1.5:** The ingestion pipeline follows a sequential pattern: load → split → embed → store. What are the trade-offs of this synchronous pipeline versus a message-queue-based asynchronous pipeline?

---

## 2. FastAPI & API Design

**Q2.1:** The API uses Pydantic models for both request validation (`IngestRequest`, `SearchRequest`) and response serialization (`IngestResponse`, `SearchResponse`). What advantages does this provide over raw dictionaries?

**Q2.2:** Explain the role of `asyncio.to_thread()` in the ingestion and search services. Why is it used, and what would happen if CPU-bound tasks ran directly on the event loop?

**Q2.3:** The ingest endpoint returns HTTP 200 with a `"status": "failure"` body for document-level errors (e.g., file not found) instead of returning HTTP 4xx. What are the pros and cons of this approach?

**Q2.4:** How does the application handle CORS, and in what scenario would the current `allow_origins=["*"]` configuration be a security concern?

---

## 3. NLP & Sentence Processing

**Q3.1:** Why was NLTK's Punkt tokenizer chosen for sentence segmentation? What types of edge cases does it handle well (e.g., abbreviations like "Dr.", decimal numbers)?

**Q3.2:** How are sentence IDs generated? Explain the format `{document_id}::{sentence_index}::{content_hash[:8]}` and why each component is included.

**Q3.3:** The `SentenceSplitter` tracks `start_offset` and `end_offset` for each sentence. What practical feature could these offsets enable in a production application?

**Q3.4:** What would happen if you replaced NLTK Punkt with a simple regex-based splitter (e.g., splitting on `.!?`)? Give specific examples where results would differ.

---

## 4. Embeddings

**Q4.1:** The project uses OpenAI's `text-embedding-3-small` model, which produces 1536-dimensional vectors. What does each dimension conceptually represent, and why are high-dimensional vectors useful for semantic search?

**Q4.2:** What is the maximum token limit for `text-embedding-3-small`, and how might the system fail if a sentence exceeds this limit? How would you mitigate this?

**Q4.3:** If you wanted to eliminate the dependency on OpenAI's API (e.g., for cost or privacy reasons), what alternative embedding approaches could you use? What trade-offs would each involve?

**Q4.4:** The `EmbeddingsService` wraps LangChain's `OpenAIEmbeddings`. What value does LangChain provide here versus calling the OpenAI API directly?

---

## 5. Vector Database & Similarity Search

**Q5.1:** Explain what HNSW (Hierarchical Navigable Small World) indexing is and why ChromaDB uses it. How does it differ from brute-force nearest-neighbor search?

**Q5.2:** The system uses cosine similarity as its distance metric. Why is cosine similarity preferred over Euclidean distance for comparing text embeddings? When might Euclidean distance be more appropriate?

**Q5.3:** What does a cosine similarity score of 0.85 mean in practical terms? How would you determine a threshold for "relevant" vs. "irrelevant" results?

**Q5.4:** ChromaDB is configured with persistent local storage (`./data/chroma_db`). What are the limitations of this approach compared to running ChromaDB in client-server mode or using a managed vector database like Pinecone?

**Q5.5:** If the database contained 10 million sentences, how would search performance be affected? What strategies could you use to maintain low-latency queries at scale?

---

## 6. PDF Processing & Document Management

**Q6.1:** How does the `DocumentProcessor` generate a stable document ID? Why is content-based hashing used instead of, for example, the file name alone?

**Q6.2:** What are the current limitations of PDF processing in this system? Name at least three types of PDF content that would not be handled correctly.

**Q6.3:** If a user re-ingests the same PDF, what happens? How does the system handle or not handle duplicate detection?

---

## 7. Error Handling

**Q7.1:** The project defines a custom error hierarchy: `BaseAppError` → `DocumentLoadError`, `IngestionError`, `SearchError`. Why create custom exceptions instead of using Python's built-in exceptions?

**Q7.2:** In the ingest route, `DocumentLoadError` results in a 200 response with failure status, while unexpected exceptions result in a 500 response. Justify this design choice.

**Q7.3:** How does the `SearchService` wrap errors? What information is preserved when a low-level exception is caught and re-raised as a `SearchError`?

---

## 8. Testing

**Q8.1:** The test suite uses `AsyncMock` and `MagicMock` extensively. Explain the difference between these two and when you would use each.

**Q8.2:** How do the tests verify that document IDs and sentence IDs are stable (i.e., deterministic)? Why is ID stability important for this system?

**Q8.3:** The API route tests use FastAPI's `TestClient`. How does this differ from making actual HTTP requests, and what are the implications for test reliability?

**Q8.4:** If you were asked to add an integration test that exercises the full ingestion pipeline (PDF → sentences → embeddings → ChromaDB), how would you set it up without requiring an OpenAI API key?

---

## 9. DevOps & Configuration

**Q9.1:** The Dockerfile uses `python:3.11-slim` as its base image and pre-downloads the NLTK Punkt tokenizer during the build. Why is the tokenizer downloaded at build time rather than at runtime?

**Q9.2:** How does the application manage configuration (API keys, paths, model names)? What is the role of `pydantic-settings`, and how does it improve over using `os.environ` directly?

**Q9.3:** What would you change in the Docker setup to make this application production-ready? Consider security, performance, and observability.

---

## 10. Scalability & Production Readiness

**Q10.1:** Name three changes you would make before deploying this service to handle 1,000 concurrent users.

**Q10.2:** The current system has no authentication or rate limiting. Describe how you would add API key-based authentication and per-key rate limiting to the FastAPI application.

**Q10.3:** How would you implement incremental indexing so that only new or modified documents are re-processed, rather than re-ingesting everything?

**Q10.4:** If you needed to support multi-language documents, what components of the system would need to change and why?
