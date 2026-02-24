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
<<<<<<< HEAD


class SentenceMatch(BaseModel):
    """A single sentence result in a search response."""

    sentence_id: str
    sentence_text: str
    document_id: str
    sentence_index: int
    score: float


class SearchResponse(BaseModel):
    """Response body for POST /api/v1/search."""

    query: str
    top_matches: List[SentenceMatch]
    time_elapsed_seconds: float
=======
>>>>>>> 2cb1d98 (Add ingest endpoint)
