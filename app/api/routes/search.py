"""POST /api/v1/search route."""

import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.models.request import SearchRequest
from app.api.models.response import SearchResponse, SentenceMatch
from app.errors.search_errors import SearchError
from app.services.search_service import SearchService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def get_search_service() -> SearchService:
    """Dependency provider for SearchService."""
    return SearchService()


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Perform semantic search over indexed sentences.

    Returns sentences most similar to the query using vector similarity.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    start = time.monotonic()

    try:
        result = await service.search(query=request.query, k=request.k)
    except SearchError as exc:
        logger.error("Search pipeline error: %s", exc.message, exc_info=True)
        raise HTTPException(status_code=500, detail=exc.message) from exc
    except Exception as exc:
        logger.error("Unexpected error during search", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    elapsed = round(time.monotonic() - start, 4)

    return SearchResponse(
        query=result.query,
        top_matches=[
            SentenceMatch(
                sentence_id=m.sentence_id,
                sentence_text=m.sentence_text,
                document_id=m.document_id,
                sentence_index=m.sentence_index,
                score=m.score,
            )
            for m in result.top_matches
        ],
        time_elapsed_seconds=elapsed,
    )
