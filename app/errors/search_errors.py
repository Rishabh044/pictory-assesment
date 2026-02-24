"""Search-specific error classes."""

from app.errors.base import BaseAppError


class SearchError(BaseAppError):
    """Raised when the search pipeline fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        details = {"cause": str(cause)} if cause else {}
        super().__init__(message=message, details=details)
        self.cause = cause
