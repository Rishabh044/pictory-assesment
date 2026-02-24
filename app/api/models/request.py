"""Request schemas for the API."""

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for POST /api/v1/ingest."""

    file_path: str = Field(
        ...,
        description="Path to the PDF file to ingest.",
        examples=["./data/documents/sample.pdf"],
    )

