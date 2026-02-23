"""Tests for the VectorStoreManager."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.core.embeddings import EmbeddingsService
from app.core.vector_store import VectorStoreManager


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(openai_api_key="test-key")


@pytest.fixture
def mock_embeddings_service():
    """Create a mock EmbeddingsService."""
    service = MagicMock(spec=EmbeddingsService)
    service.get_embeddings_instance.return_value = MagicMock()
    return service


@pytest.fixture
def sample_documents():
    """Three Document objects with full sentence metadata."""
    return [
        Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={
                "document_id": "test_doc.txt",
                "sentence_id": "test_doc.txt::0::a1b2c3d4",
                "sentence_text": "The quick brown fox jumps over the lazy dog.",
                "sentence_index": 0,
                "start_offset": 0,
                "end_offset": 44,
            },
        ),
        Document(
            page_content="A fast animal leaped across the sleeping hound.",
            metadata={
                "document_id": "test_doc.txt",
                "sentence_id": "test_doc.txt::1::e5f6g7h8",
                "sentence_text": "A fast animal leaped across the sleeping hound.",
                "sentence_index": 1,
                "start_offset": 45,
                "end_offset": 92,
            },
        ),
        Document(
            page_content="Semantic search enables finding related sentences.",
            metadata={
                "document_id": "test_doc.txt",
                "sentence_id": "test_doc.txt::2::i9j0k1l2",
                "sentence_text": "Semantic search enables finding related sentences.",
                "sentence_index": 2,
                "start_offset": 93,
                "end_offset": 143,
            },
        ),
    ]


@pytest.fixture
def mock_chroma(mock_settings, mock_embeddings_service):
    """Provide a VectorStoreManager with a mocked Chroma instance."""
    with patch("app.core.vector_store.Chroma") as MockChroma:
        chroma_instance = MagicMock()
        MockChroma.return_value = chroma_instance
        manager = VectorStoreManager(
            embeddings_service=mock_embeddings_service,
            settings=mock_settings,
        )
        yield manager, chroma_instance, MockChroma


class TestVectorStoreManager:
    """Tests for VectorStoreManager class."""

    def test_add_documents_returns_count(self, mock_chroma, sample_documents):
        """add_documents returns the count of documents added."""
        manager, chroma_instance, _ = mock_chroma
        chroma_instance.add_documents.return_value = None

        result = manager.add_documents(sample_documents)

        assert result == len(sample_documents)

    def test_add_documents_delegates_to_chroma(self, mock_chroma, sample_documents):
        """add_documents calls chroma.add_documents with the correct docs."""
        manager, chroma_instance, _ = mock_chroma

        manager.add_documents(sample_documents)

        chroma_instance.add_documents.assert_called_once_with(sample_documents)

    def test_add_documents_empty_list(self, mock_chroma):
        """add_documents returns 0 for empty input without calling Chroma."""
        manager, chroma_instance, _ = mock_chroma

        result = manager.add_documents([])

        assert result == 0
        chroma_instance.add_documents.assert_not_called()

    def test_similarity_search_returns_results(self, mock_chroma, sample_documents):
        """similarity_search_with_score returns (Document, float) tuples."""
        manager, chroma_instance, _ = mock_chroma
        expected = [(sample_documents[0], 0.95), (sample_documents[1], 0.80)]
        chroma_instance.similarity_search_with_score.return_value = expected

        result = manager.similarity_search_with_score("fox jumping")

        assert result == expected
        assert all(isinstance(doc, Document) and isinstance(score, float) for doc, score in result)

    def test_similarity_search_passes_k(self, mock_chroma, sample_documents):
        """k parameter is forwarded to Chroma."""
        manager, chroma_instance, _ = mock_chroma
        chroma_instance.similarity_search_with_score.return_value = []

        manager.similarity_search_with_score("test query", k=10)

        chroma_instance.similarity_search_with_score.assert_called_once_with(
            "test query", k=10, filter=None
        )

    def test_similarity_search_passes_filter(self, mock_chroma, sample_documents):
        """filter dict is forwarded to Chroma."""
        manager, chroma_instance, _ = mock_chroma
        chroma_instance.similarity_search_with_score.return_value = []
        filter_dict = {"document_id": "test_doc.txt"}

        manager.similarity_search_with_score("test query", filter=filter_dict)

        chroma_instance.similarity_search_with_score.assert_called_once_with(
            "test query", k=5, filter=filter_dict
        )

    def test_similarity_search_no_filter(self, mock_chroma):
        """filter=None works without raising an exception."""
        manager, chroma_instance, _ = mock_chroma
        chroma_instance.similarity_search_with_score.return_value = []

        result = manager.similarity_search_with_score("test query", filter=None)

        assert result == []

    def test_get_similar_excludes_self(self, mock_chroma, sample_documents):
        """get_similar_to_document removes the source document from results."""
        manager, chroma_instance, _ = mock_chroma
        source = sample_documents[0]
        # Chroma returns source + two others
        chroma_instance.similarity_search_with_score.return_value = [
            (sample_documents[0], 1.0),
            (sample_documents[1], 0.85),
            (sample_documents[2], 0.70),
        ]

        result = manager.get_similar_to_document(source, k=5)

        returned_ids = [doc.metadata["sentence_id"] for doc, _ in result]
        assert source.metadata["sentence_id"] not in returned_ids

    def test_get_similar_returns_up_to_k(self, mock_chroma, sample_documents):
        """Result count is capped at k."""
        manager, chroma_instance, _ = mock_chroma
        # Return 3 docs (including self) from Chroma; k=1 means only 1 non-self result
        chroma_instance.similarity_search_with_score.return_value = [
            (sample_documents[0], 1.0),
            (sample_documents[1], 0.85),
            (sample_documents[2], 0.70),
        ]

        result = manager.get_similar_to_document(sample_documents[0], k=1)

        assert len(result) <= 1

    def test_get_similar_uses_document_text_as_query(self, mock_chroma, sample_documents):
        """Chroma is queried with the document's page_content."""
        manager, chroma_instance, _ = mock_chroma
        chroma_instance.similarity_search_with_score.return_value = []
        source = sample_documents[0]

        manager.get_similar_to_document(source, k=3)

        call_args = chroma_instance.similarity_search_with_score.call_args
        assert call_args[0][0] == source.page_content

    def test_chroma_initialized_with_persist_directory(self, mock_settings, mock_embeddings_service):
        """Constructor passes persist_directory from settings to Chroma."""
        with patch("app.core.vector_store.Chroma") as MockChroma:
            MockChroma.return_value = MagicMock()

            VectorStoreManager(
                embeddings_service=mock_embeddings_service,
                settings=mock_settings,
            )

            _, kwargs = MockChroma.call_args
            assert kwargs["persist_directory"] == mock_settings.chroma_persist_directory
