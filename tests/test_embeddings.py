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

    
