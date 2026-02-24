"""Search service: orchestrates vector similarity search pipeline."""

import asyncio
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from app.core.vector_store import VectorStoreManager
from app.errors.search_errors import SearchError
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentenceResult:
    sentence_id: str
    sentence_text: str
    document_id: str
    sentence_index: int
    score: float


@dataclass
class SearchResult:
    query: str
    top_matches: List[SentenceResult] = field(default_factory=list)


def _doc_to_result(doc: Document, score: float) -> SentenceResult:
    """Extract metadata fields from a LangChain Document into a SentenceResult."""
    meta = doc.metadata
    return SentenceResult(
        sentence_id=meta.get("sentence_id", ""),
        sentence_text=meta.get("sentence_text", doc.page_content),
        document_id=meta.get("document_id", ""),
        sentence_index=meta.get("sentence_index", 0),
        score=score,
    )


class SearchService:
    """Orchestrates semantic search using the vector store."""

    def __init__(self, vector_store: VectorStoreManager | None = None):
        self._vector_store = vector_store or VectorStoreManager()

    async def search(self, query: str, k: int = 5) -> SearchResult:
        """
        Search for sentences semantically similar to the query.

        Args:
            query: Plain-text search query.
            k: Number of top matches to return.

        Returns:
            SearchResult with top_matches populated.

        Raises:
            SearchError: If the vector store query fails.
        """
        logger.info("Starting search: query='%s', k=%d", query, k)

        try:
            raw_matches = await asyncio.to_thread(
                self._vector_store.similarity_search_with_score, query, k
            )
        except Exception as exc:
            raise SearchError(
                message="Vector store query failed", cause=exc
            ) from exc

        top_matches = [_doc_to_result(doc, score) for doc, score in raw_matches]

        logger.info("Search returned %d matches", len(top_matches))

        return SearchResult(query=query, top_matches=top_matches)
