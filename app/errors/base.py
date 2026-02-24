"""Base error classes for the semantic search system."""


class BaseAppError(Exception):
    """Base class for all application-specific errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"
