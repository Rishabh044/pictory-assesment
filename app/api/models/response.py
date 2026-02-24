"""Response schemas for the API."""

from typing import List, Union

from pydantic import BaseModel


class DocumentDetail(BaseModel):
    """Per-document detail in a successful ingest response."""

    document_id: str
    sentences_count: int
    status: str


class IngestResponse(BaseModel):
    """Response body for POST /api/v1/ingest."""

    status: str
    sentences_indexed: int
    time_elapsed_seconds: float
    details: Union[List[DocumentDetail], str]
