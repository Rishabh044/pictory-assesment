"""Embeddings service wrapper for OpenAI embeddings."""

from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config import Settings, get_settings


class EmbeddingsService:
    """Wrapper around LangChain's OpenAIEmbeddings."""

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the embeddings service.

        Args:
            settings: Optional Settings instance. If not provided, loads from environment.
        """
        self._settings = settings or get_settings()
        self._embeddings = OpenAIEmbeddings(
            model=self._settings.embedding_model,
            openai_api_key=self._settings.openai_api_key,
        )

    def get_embeddings_instance(self) -> OpenAIEmbeddings:
        """
        Get the underlying OpenAIEmbeddings instance.

        Returns:
            The LangChain OpenAIEmbeddings object for use with Chroma.
        """
        return self._embeddings
