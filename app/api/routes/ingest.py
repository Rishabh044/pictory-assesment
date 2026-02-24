"""POST /api/v1/ingest route."""

import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.models.request import IngestRequest
from app.api.models.response import DocumentDetail, IngestResponse
from app.errors.ingestion_errors import DocumentLoadError, IngestionError
from app.services.ingestion_service import IngestionService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def get_ingestion_service() -> IngestionService:
    """Dependency provider for IngestionService."""
    return IngestionService()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """
    Ingest a PDF document into the semantic search index.

    Splits the document into sentences, embeds them with OpenAI, and stores
    them in the Chroma vector store.
    """
    if not request.file_path.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Provide a path ending in .pdf.",
        )

    start = time.monotonic()

    try:
        result = await service.ingest(file_path=request.file_path)
    except DocumentLoadError as exc:
        elapsed = round(time.monotonic() - start, 2)
        logger.warning("Document load error: %s", exc.message)
        return IngestResponse(
            status="failure",
            sentences_indexed=0,
            time_elapsed_seconds=elapsed,
            details=exc.message,
        )
    except IngestionError as exc:
        elapsed = round(time.monotonic() - start, 2)
        logger.error("Ingestion pipeline error: %s", exc.message, exc_info=True)
        return IngestResponse(
            status="failure",
            sentences_indexed=0,
            time_elapsed_seconds=elapsed,
            details=exc.message,
        )
    except Exception as exc:
        logger.error("Unexpected error during ingestion", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    elapsed = round(time.monotonic() - start, 2)

    return IngestResponse(
        status="success",
        sentences_indexed=result.sentences_count,
        time_elapsed_seconds=elapsed,
        details=[
            DocumentDetail(
                document_id=result.document_id,
                sentences_count=result.sentences_count,
                status="success",
            )
        ],
    )
