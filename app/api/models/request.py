"""Request schemas for the API."""

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for POST /api/v1/ingest."""

    file_path: str = Field(
        ...,
        description="Path to the PDF file to ingest.",
        examples=["./data/documents/sample.pdf"],
    )

class SearchRequest(BaseModel):
    """Request body for POST /api/v1/search."""

    query: str = Field(..., description="Plain-text search query.")
    k: int = Field(default=5, ge=1, le=50, description="Results per search phase.")


