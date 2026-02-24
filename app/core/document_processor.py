"""Document processor for loading and preparing PDFs for ingestion."""

import hashlib
from pathlib import Path
from typing import Tuple

from langchain_community.document_loaders import PyPDFLoader

from app.errors.ingestion_errors import DocumentLoadError
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf"}


class DocumentProcessor:
    """Loads PDF documents from disk and produces (document_id, text) pairs."""

    def load_pdf(self, file_path: str) -> Tuple[str, str]:
        """
        Load a PDF file and return its document_id and full text.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Tuple of (document_id, concatenated_text).

        Raises:
            DocumentLoadError: If the file does not exist, is not a PDF, or cannot be parsed.
        """
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoadError(file_path=file_path, reason="File does not exist")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                file_path=file_path,
                reason=f"Unsupported file type '{path.suffix}'. Only PDF files are accepted.",
            )

        try:
            loader = PyPDFLoader(str(path))
            pages = loader.load()
        except Exception as exc:
            raise DocumentLoadError(
                file_path=file_path,
                reason=f"PyPDFLoader failed: {exc}",
            ) from exc

        if not pages:
            raise DocumentLoadError(
                file_path=file_path,
                reason="PDF produced no pages",
            )

        full_text = "\n\n".join(
            page.page_content for page in pages if page.page_content
        )

        if not full_text.strip():
            raise DocumentLoadError(
                file_path=file_path,
                reason="PDF contains no extractable text content",
            )

        content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:8]
        document_id = f"{path.stem}_{content_hash}"

        logger.info(
            "Loaded PDF '%s' → document_id='%s' (%d chars)",
            file_path,
            document_id,
            len(full_text),
        )

        return document_id, full_text
