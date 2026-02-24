"""Ingestion service: orchestrates load → split → embed → store pipeline."""

import asyncio
import time
from dataclasses import dataclass

from app.core.document_processor import DocumentProcessor
from app.core.sentence_splitter import SentenceSplitter
from app.core.vector_store import VectorStoreManager
from app.errors.ingestion_errors import DocumentLoadError, IngestionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    document_id: str
    sentences_count: int
    status: str


class IngestionService:
    """Orchestrates the full document ingestion pipeline."""

    def __init__(
        self,
        document_processor: DocumentProcessor | None = None,
        sentence_splitter: SentenceSplitter | None = None,
        vector_store: VectorStoreManager | None = None,
    ):
        self._processor = document_processor or DocumentProcessor()
        self._splitter = sentence_splitter or SentenceSplitter()
        self._vector_store = vector_store or VectorStoreManager()

    async def ingest(self, file_path: str) -> IngestionResult:
        """
        Ingest a PDF file into the vector store.

        Args:
            file_path: Path to the PDF file.
            regenerate: If True, delete existing sentences for this document first.

        Returns:
            IngestionResult with outcome details.

        Raises:
            DocumentLoadError: If the file cannot be loaded.
            IngestionError: If embedding or storage fails.
        """
        logger.info("Starting ingestion: file_path='%s'", file_path)

        # Step 1: Load document
        document_id, text = await asyncio.to_thread(
            self._processor.load_pdf, file_path
        )
        
        # Step 2: Split into sentences
        sentences = await asyncio.to_thread(
            self._splitter.split_text, text, document_id
        )

        if not sentences:
            raise IngestionError(
                message=f"Document '{file_path}' produced no sentences after splitting."
            )

        logger.info("Split into %d sentences for document_id='%s'", len(sentences), document_id)

        # Step 4: Embed and store
        try:
            count = await asyncio.to_thread(
                self._vector_store.add_documents, sentences
            )
        except Exception as exc:
            raise IngestionError(
                message="Failed to store embeddings in vector store",
                cause=exc,
            ) from exc

        logger.info("Stored %d sentences for document_id='%s'", count, document_id)

        return IngestionResult(
            document_id=document_id,
            sentences_count=count,
            status="success",
        )
