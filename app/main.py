"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.ingest import router as ingest_router
<<<<<<< HEAD
from app.api.routes.search import router as search_router
=======
>>>>>>> 2cb1d98 (Add ingest endpoint)

app = FastAPI(
    title="Semantic Sentence Search API",
    description="Ingest documents and search them semantically using OpenAI embeddings and Chroma.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/api/v1")
<<<<<<< HEAD
app.include_router(search_router, prefix="/api/v1")
=======
>>>>>>> 2cb1d98 (Add ingest endpoint)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
