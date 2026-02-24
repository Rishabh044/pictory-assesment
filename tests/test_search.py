"""Tests for the search pipeline: SearchService and search route."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.routes.search import get_search_service
from app.errors.search_errors import SearchError
from app.main import app
from app.services.search_service import SearchResult, SearchService, SentenceResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Synchronous TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_docs():
    """Two LangChain Documents with sentence metadata."""
    return [
        Document(
            page_content="The quick brown fox jumps.",
            metadata={
                "document_id": "doc_abc12345",
                "sentence_id": "doc_abc12345::0::aabbccdd",
                "sentence_text": "The quick brown fox jumps.",
                "sentence_index": 0,
                "start_offset": 0,
                "end_offset": 25,
            },
        ),
        Document(
            page_content="Lazy dogs sleep all day.",
            metadata={
                "document_id": "doc_abc12345",
                "sentence_id": "doc_abc12345::1::eeff0011",
                "sentence_text": "Lazy dogs sleep all day.",
                "sentence_index": 1,
                "start_offset": 26,
                "end_offset": 49,
            },
        ),
    ]


# ---------------------------------------------------------------------------
# TestSearchService
# ---------------------------------------------------------------------------


class TestSearchService:
    """Unit tests for SearchService.search()."""

    @pytest.fixture
    def mock_vector_store(self, sample_docs):
        store = MagicMock()
        store.similarity_search_with_score.return_value = [
            (sample_docs[0], 0.1),
            (sample_docs[1], 0.3),
        ]
        return store

    @pytest.fixture
    def service(self, mock_vector_store):
        return SearchService(vector_store=mock_vector_store)

    @pytest.mark.asyncio
    async def test_search_returns_search_result(self, service):
        result = await service.search("fox", k=5)
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_populates_top_matches(self, service):
        result = await service.search("fox", k=5)
        assert len(result.top_matches) == 2

    @pytest.mark.asyncio
    async def test_search_result_contains_query(self, service):
        result = await service.search("fox", k=5)
        assert result.query == "fox"

    @pytest.mark.asyncio
    async def test_search_top_match_shape(self, service):
        result = await service.search("fox", k=5)
        match = result.top_matches[0]
        assert isinstance(match, SentenceResult)
        assert match.sentence_id == "doc_abc12345::0::aabbccdd"
        assert match.sentence_text == "The quick brown fox jumps."
        assert match.document_id == "doc_abc12345"
        assert match.sentence_index == 0
        assert match.score == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_search_passes_k_to_vector_store(self, service, mock_vector_store):
        await service.search("fox", k=10)
        mock_vector_store.similarity_search_with_score.assert_called_once_with("fox", 10)

    @pytest.mark.asyncio
    async def test_search_empty_results_returns_empty_lists(self, mock_vector_store):
        mock_vector_store.similarity_search_with_score.return_value = []
        service = SearchService(vector_store=mock_vector_store)
        result = await service.search("nothing", k=5)
        assert result.top_matches == []

    @pytest.mark.asyncio
    async def test_search_raises_search_error_on_store_failure(self, mock_vector_store):
        mock_vector_store.similarity_search_with_score.side_effect = RuntimeError("store down")
        service = SearchService(vector_store=mock_vector_store)
        with pytest.raises(SearchError):
            await service.search("query", k=5)

    @pytest.mark.asyncio
    async def test_search_error_wraps_original_cause(self, mock_vector_store):
        cause = RuntimeError("store down")
        mock_vector_store.similarity_search_with_score.side_effect = cause
        service = SearchService(vector_store=mock_vector_store)
        with pytest.raises(SearchError) as exc_info:
            await service.search("query", k=5)
        assert exc_info.value.cause is cause


# ---------------------------------------------------------------------------
# TestSearchRoute
# ---------------------------------------------------------------------------


class TestSearchRoute:
    """Integration tests for POST /api/v1/search."""

    def _make_mock_service(self, result=None, side_effect=None):
        """Build a mock SearchService with controlled search() behaviour."""
        mock_svc = MagicMock(spec=SearchService)
        if side_effect:
            mock_svc.search = AsyncMock(side_effect=side_effect)
        else:
            mock_svc.search = AsyncMock(return_value=result)
        return mock_svc

    @pytest.fixture(autouse=True)
    def clear_overrides(self):
        """Ensure dependency overrides are cleaned up after each test."""
        yield
        app.dependency_overrides.clear()

    def _success_result(self, query="test query"):
        return SearchResult(
            query=query,
            top_matches=[
                SentenceResult(
                    sentence_id="doc_abc::0::aabbccdd",
                    sentence_text="A sample sentence.",
                    document_id="doc_abc",
                    sentence_index=0,
                    score=0.05,
                )
            ],
        )

    def test_valid_query_returns_200(self, client):
        result = self._success_result()
        app.dependency_overrides[get_search_service] = lambda: self._make_mock_service(result=result)
        resp = client.post("/api/v1/search", json={"query": "test query"})
        assert resp.status_code == 200

    def test_response_shape(self, client):
        result = self._success_result("test query")
        app.dependency_overrides[get_search_service] = lambda: self._make_mock_service(result=result)
        resp = client.post("/api/v1/search", json={"query": "test query"})
        body = resp.json()
        assert body["query"] == "test query"
        assert isinstance(body["top_matches"], list)
        assert "time_elapsed_seconds" in body
        assert isinstance(body["time_elapsed_seconds"], float)

    def test_sentence_match_shape(self, client):
        result = self._success_result()
        app.dependency_overrides[get_search_service] = lambda: self._make_mock_service(result=result)
        resp = client.post("/api/v1/search", json={"query": "test query"})
        match = resp.json()["top_matches"][0]
        assert "sentence_id" in match
        assert "sentence_text" in match
        assert "document_id" in match
        assert "sentence_index" in match
        assert "score" in match

    def test_empty_query_returns_400(self, client):
        app.dependency_overrides[get_search_service] = lambda: MagicMock(spec=SearchService)
        resp = client.post("/api/v1/search", json={"query": ""})
        assert resp.status_code == 400

    def test_whitespace_only_query_returns_400(self, client):
        app.dependency_overrides[get_search_service] = lambda: MagicMock(spec=SearchService)
        resp = client.post("/api/v1/search", json={"query": "   "})
        assert resp.status_code == 400

    def test_k_zero_returns_422(self, client):
        app.dependency_overrides[get_search_service] = lambda: MagicMock(spec=SearchService)
        resp = client.post("/api/v1/search", json={"query": "test", "k": 0})
        assert resp.status_code == 422

    def test_k_above_max_returns_422(self, client):
        app.dependency_overrides[get_search_service] = lambda: MagicMock(spec=SearchService)
        resp = client.post("/api/v1/search", json={"query": "test", "k": 51})
        assert resp.status_code == 422

    def test_empty_matches_returns_200_with_empty_list(self, client):
        result = SearchResult(query="nothing", top_matches=[])
        app.dependency_overrides[get_search_service] = lambda: self._make_mock_service(result=result)
        resp = client.post("/api/v1/search", json={"query": "nothing"})
        assert resp.status_code == 200
        assert resp.json()["top_matches"] == []

    def test_search_error_returns_500(self, client):
        error = SearchError(message="Vector store unavailable")
        app.dependency_overrides[get_search_service] = lambda: self._make_mock_service(side_effect=error)
        resp = client.post("/api/v1/search", json={"query": "test"})
        assert resp.status_code == 500

    def test_unexpected_exception_returns_500(self, client):
        app.dependency_overrides[get_search_service] = lambda: self._make_mock_service(
            side_effect=RuntimeError("boom")
        )
        resp = client.post("/api/v1/search", json={"query": "test"})
        assert resp.status_code == 500

    def test_default_k_is_five(self, client):
        result = self._success_result()
        mock_svc = self._make_mock_service(result=result)
        app.dependency_overrides[get_search_service] = lambda: mock_svc
        client.post("/api/v1/search", json={"query": "test"})
        mock_svc.search.assert_called_once_with(query="test", k=5)

    def test_custom_k_forwarded_to_service(self, client):
        result = self._success_result()
        mock_svc = self._make_mock_service(result=result)
        app.dependency_overrides[get_search_service] = lambda: mock_svc
        client.post("/api/v1/search", json={"query": "test", "k": 20})
        mock_svc.search.assert_called_once_with(query="test", k=20)
