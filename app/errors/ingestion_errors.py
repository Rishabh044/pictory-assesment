"""Ingestion-specific error classes."""

from app.errors.base import BaseAppError


class DocumentLoadError(BaseAppError):
    """Raised when a document cannot be loaded from disk."""

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message=f"Failed to load document '{file_path}': {reason}",
            details={"file_path": file_path, "reason": reason},
        )
        self.file_path = file_path


class IngestionError(BaseAppError):
    """Raised when the ingestion pipeline fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        details = {"cause": str(cause)} if cause else {}
        super().__init__(message=message, details=details)
        self.cause = cause
