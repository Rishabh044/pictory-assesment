"""Chroma vector store wrapper for sentence embeddings."""

from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.core.embeddings import EmbeddingsService


class VectorStoreManager:
    """Wrapper around LangChain's Chroma vector store for sentence embeddings."""

    def __init__(
        self,
        embeddings_service: EmbeddingsService | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize the vector store.

        Args:
            embeddings_service: Optional EmbeddingsService instance. Created if not provided.
            settings: Optional Settings instance. Loaded from environment if not provided.
        """
        self._settings = settings or get_settings()
        self._embeddings_service = embeddings_service or EmbeddingsService(settings=self._settings)
        self._store = Chroma(
            collection_name="sentence_embeddings",
            embedding_function=self._embeddings_service.get_embeddings_instance(),
            persist_directory=self._settings.chroma_persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects with page_content and metadata.

        Returns:
            Number of documents added.
        """
        if not documents:
            return 0
        self._store.add_documents(documents)
        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query string.

        Args:
            query: Plain text query string.
            k: Number of results to return.
            filter: Optional metadata filter dict.

        Returns:
            List of (Document, score) tuples ordered by similarity.
        """
        return self._store.similarity_search_with_score(query, k=k, filter=filter)

    def get_similar_to_document(
        self,
        document: Document,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Find sentences similar to a given document, excluding the document itself.

        Args:
            document: Source Document to find similar sentences for.
            k: Maximum number of results to return.

        Returns:
            List of (Document, score) tuples, self excluded, capped at k.
        """
        source_id = document.metadata.get("sentence_id")
        results = self._store.similarity_search_with_score(
            document.page_content, k=k + 1, filter=None
        )
        filtered = [
            (doc, score)
            for doc, score in results
            if doc.metadata.get("sentence_id") != source_id
        ]
        return filtered[:k]
