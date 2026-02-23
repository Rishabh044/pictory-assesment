"""Tests for the embeddings service."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.core.embeddings import EmbeddingsService


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        openai_api_key="test-api-key",
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture
def mock_embedding_vector():
    """Create a mock embedding vector with 1536 dimensions."""
    return [0.1] * 1536


class TestEmbeddingsService:
    """Tests for EmbeddingsService class."""

    @patch.object(OpenAIEmbeddings, "embed_query")
    def test_embed_text_returns_list(self, mock_embed_query, mock_settings, mock_embedding_vector):
        """Single text returns list of floats."""
        mock_embed_query.return_value = mock_embedding_vector

        service = EmbeddingsService(settings=mock_settings)
        result = service.embed_text("Hello world")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        mock_embed_query.assert_called_once_with("Hello world")

    @patch.object(OpenAIEmbeddings, "embed_documents")
    def test_embed_texts_batch(self, mock_embed_documents, mock_settings, mock_embedding_vector):
        """Multiple texts return correct number of embeddings."""
        texts = ["First text", "Second text", "Third text"]
        mock_embed_documents.return_value = [mock_embedding_vector] * 3

        service = EmbeddingsService(settings=mock_settings)
        result = service.embed_texts(texts)

        assert len(result) == 3
        assert all(isinstance(emb, list) for emb in result)
        mock_embed_documents.assert_called_once_with(texts)

    @patch.object(OpenAIEmbeddings, "embed_query")
    def test_embedding_dimension(self, mock_embed_query, mock_settings, mock_embedding_vector):
        """Verify embedding vector dimension is 1536."""
        mock_embed_query.return_value = mock_embedding_vector

        service = EmbeddingsService(settings=mock_settings)
        result = service.embed_text("Test text")

        assert len(result) == 1536

    @patch.object(OpenAIEmbeddings, "embed_query")
    def test_empty_text_handling(self, mock_embed_query, mock_settings, mock_embedding_vector):
        """Empty string is passed to the embeddings model."""
        mock_embed_query.return_value = mock_embedding_vector

        service = EmbeddingsService(settings=mock_settings)
        result = service.embed_text("")

        mock_embed_query.assert_called_once_with("")
        assert isinstance(result, list)

    def test_get_embeddings_instance_returns_instance(self, mock_settings):
        """Returns OpenAIEmbeddings instance."""
        service = EmbeddingsService(settings=mock_settings)
        embeddings = service.get_embeddings_instance()

        assert isinstance(embeddings, OpenAIEmbeddings)

    def test_uses_correct_model(self, mock_settings):
        """Verifies text-embedding-3-small model is used."""
        service = EmbeddingsService(settings=mock_settings)
        embeddings = service.get_embeddings_instance()

        assert embeddings.model == "text-embedding-3-small"

    def test_settings_passed_to_embeddings(self, mock_settings):
        """Verify settings are correctly passed to OpenAIEmbeddings."""
        service = EmbeddingsService(settings=mock_settings)
        embeddings = service.get_embeddings_instance()

        assert embeddings.openai_api_key.get_secret_value() == "test-api-key"

    @patch.object(OpenAIEmbeddings, "embed_documents")
    def test_embed_texts_empty_list(self, mock_embed_documents, mock_settings):
        """Empty list returns empty list."""
        mock_embed_documents.return_value = []

        service = EmbeddingsService(settings=mock_settings)
        result = service.embed_texts([])

        assert result == []
        mock_embed_documents.assert_called_once_with([])
