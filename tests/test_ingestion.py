"""Tests for the ingestion pipeline: DocumentProcessor, IngestionService, and ingest route."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.models.response import IngestResponse
from app.api.routes.ingest import get_ingestion_service
from app.config import Settings
from app.core.document_processor import DocumentProcessor
from app.errors.ingestion_errors import DocumentLoadError, IngestionError
from app.main import app
from app.services.ingestion_service import IngestionResult, IngestionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Synchronous TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a dummy .pdf file on disk (content does not need to be valid PDF)."""
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake pdf content")
    return str(pdf)


@pytest.fixture
def sample_documents():
    """A small list of LangChain Documents representing split sentences."""
    return [
        Document(
            page_content="First sentence of the document.",
            metadata={
                "document_id": "sample_abc12345",
                "sentence_id": "sample_abc12345::0::aaaabbbb",
                "sentence_text": "First sentence of the document.",
                "sentence_index": 0,
                "start_offset": 0,
                "end_offset": 31,
            },
        ),
        Document(
            page_content="Second sentence follows here.",
            metadata={
                "document_id": "sample_abc12345",
                "sentence_id": "sample_abc12345::1::ccccdddd",
                "sentence_text": "Second sentence follows here.",
                "sentence_index": 1,
                "start_offset": 32,
                "end_offset": 60,
            },
        ),
    ]


# ---------------------------------------------------------------------------
# TestDocumentProcessor
# ---------------------------------------------------------------------------


class TestDocumentProcessor:
    """Tests for DocumentProcessor.load_pdf()."""

    def test_load_pdf_raises_on_missing_file(self):
        """Raise DocumentLoadError when file does not exist."""
        processor = DocumentProcessor()
        with pytest.raises(DocumentLoadError) as exc_info:
            processor.load_pdf("/nonexistent/path/file.pdf")
        assert "does not exist" in exc_info.value.message

    def test_load_pdf_raises_on_non_pdf(self, tmp_path):
        """Raise DocumentLoadError for non-PDF extensions."""
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("some content")
        processor = DocumentProcessor()
        with pytest.raises(DocumentLoadError) as exc_info:
            processor.load_pdf(str(txt_file))
        assert "Unsupported file type" in exc_info.value.message

    def test_load_pdf_returns_document_id_and_text(self, tmp_path):
        """Successful load returns (document_id, text) with correct format."""
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"placeholder")
        extracted_text = "Hello world. This is a test document."

        mock_page = MagicMock()
        mock_page.page_content = extracted_text

        processor = DocumentProcessor()
        with patch("app.core.document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_page]
            document_id, text = processor.load_pdf(str(pdf_path))

        assert text == extracted_text
        expected_hash = hashlib.sha256(extracted_text.encode()).hexdigest()[:8]
        assert document_id == f"report_{expected_hash}"

    def test_load_pdf_document_id_is_stable(self, tmp_path):
        """Same file content always produces the same document_id."""
        pdf_path = tmp_path / "stable.pdf"
        pdf_path.write_bytes(b"placeholder")
        extracted_text = "Stable content."

        mock_page = MagicMock()
        mock_page.page_content = extracted_text

        processor = DocumentProcessor()
        with patch("app.core.document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_page]
            id1, _ = processor.load_pdf(str(pdf_path))
            MockLoader.return_value.load.return_value = [mock_page]
            id2, _ = processor.load_pdf(str(pdf_path))

        assert id1 == id2

    def test_load_pdf_document_id_changes_with_content(self, tmp_path):
        """Different content produces a different document_id."""
        pdf_path = tmp_path / "changing.pdf"
        pdf_path.write_bytes(b"placeholder")

        page_a = MagicMock()
        page_a.page_content = "Content version A."
        page_b = MagicMock()
        page_b.page_content = "Content version B."

        processor = DocumentProcessor()
        with patch("app.core.document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [page_a]
            id_a, _ = processor.load_pdf(str(pdf_path))

            MockLoader.return_value.load.return_value = [page_b]
            id_b, _ = processor.load_pdf(str(pdf_path))

        assert id_a != id_b

    def test_load_pdf_raises_on_empty_content(self, tmp_path):
        """Raise DocumentLoadError when PDF has no extractable text."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"placeholder")

        mock_page = MagicMock()
        mock_page.page_content = "   "

        processor = DocumentProcessor()
        with patch("app.core.document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_page]
            with pytest.raises(DocumentLoadError) as exc_info:
                processor.load_pdf(str(pdf_path))

        assert "no extractable text" in exc_info.value.message

    def test_load_pdf_raises_on_loader_exception(self, tmp_path):
        """Wrap PyPDFLoader exceptions in DocumentLoadError."""
        pdf_path = tmp_path / "bad.pdf"
        pdf_path.write_bytes(b"placeholder")

        processor = DocumentProcessor()
        with patch("app.core.document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = RuntimeError("corrupt pdf")
            with pytest.raises(DocumentLoadError) as exc_info:
                processor.load_pdf(str(pdf_path))

        assert "PyPDFLoader failed" in exc_info.value.message

    def test_load_pdf_joins_multiple_pages(self, tmp_path):
        """Text from multiple pages is joined with double newlines."""
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"placeholder")

        page1 = MagicMock()
        page1.page_content = "Page one text."
        page2 = MagicMock()
        page2.page_content = "Page two text."

        processor = DocumentProcessor()
        with patch("app.core.document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [page1, page2]
            _, text = processor.load_pdf(str(pdf_path))

        assert text == "Page one text.\n\nPage two text."


# ---------------------------------------------------------------------------
# TestIngestionService
# ---------------------------------------------------------------------------


class TestIngestionService:
    """Tests for IngestionService.ingest()."""

    @pytest.fixture
    def mock_processor(self):
        proc = MagicMock(spec=DocumentProcessor)
        proc.load_pdf.return_value = ("sample_abc12345", "Some document text.")
        return proc

    @pytest.fixture
    def mock_splitter(self, sample_documents):
        splitter = MagicMock()
        splitter.split_text.return_value = sample_documents
        return splitter

    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.add_documents.return_value = 2
        return store

    @pytest.fixture
    def service(self, mock_processor, mock_splitter, mock_vector_store):
        return IngestionService(
            document_processor=mock_processor,
            sentence_splitter=mock_splitter,
            vector_store=mock_vector_store,
        )

    @pytest.mark.asyncio
    async def test_ingest_success_returns_result(self, service, mock_vector_store):
        result = await service.ingest("doc.pdf")

        assert result.status == "success"
        assert result.sentences_count == 2
        assert result.document_id == "sample_abc12345"


    @pytest.mark.asyncio
    async def test_ingest_propagates_document_load_error(
        self, mock_processor, mock_splitter, mock_vector_store
    ):
        mock_processor.load_pdf.side_effect = DocumentLoadError(
            file_path="missing.pdf", reason="File does not exist"
        )
        service = IngestionService(
            document_processor=mock_processor,
            sentence_splitter=mock_splitter,
            vector_store=mock_vector_store,
        )

        with pytest.raises(DocumentLoadError):
            await service.ingest("missing.pdf")

    @pytest.mark.asyncio
    async def test_ingest_raises_ingestion_error_on_store_failure(
        self, mock_processor, mock_splitter, mock_vector_store
    ):
        mock_vector_store.add_documents.side_effect = RuntimeError("Chroma unavailable")
        service = IngestionService(
            document_processor=mock_processor,
            sentence_splitter=mock_splitter,
            vector_store=mock_vector_store,
        )

        with pytest.raises(IngestionError):
            await service.ingest("doc.pdf")

    @pytest.mark.asyncio
    async def test_ingest_raises_ingestion_error_on_empty_sentences(
        self, mock_processor, mock_splitter, mock_vector_store
    ):
        mock_splitter.split_text.return_value = []
        service = IngestionService(
            document_processor=mock_processor,
            sentence_splitter=mock_splitter,
            vector_store=mock_vector_store,
        )

        with pytest.raises(IngestionError) as exc_info:
            await service.ingest("doc.pdf")

        assert "no sentences" in exc_info.value.message


# ---------------------------------------------------------------------------
# TestIngestRoute
# ---------------------------------------------------------------------------


class TestIngestRoute:
    """Tests for POST /api/v1/ingest."""

    def _make_mock_service(self, result=None, side_effect=None):
        """Build a mock IngestionService with controlled ingest() behaviour."""
        mock_svc = MagicMock(spec=IngestionService)
        if side_effect:
            mock_svc.ingest = AsyncMock(side_effect=side_effect)
        else:
            mock_svc.ingest = AsyncMock(return_value=result)
        return mock_svc

    @pytest.fixture(autouse=True)
    def clear_overrides(self):
        """Ensure dependency overrides are cleaned up after each test."""
        yield
        app.dependency_overrides.clear()

    def test_post_ingest_success_returns_200(self, client):
        result = IngestionResult(document_id="sample_abc12345", sentences_count=42, status="success")
        app.dependency_overrides[get_ingestion_service] = lambda: self._make_mock_service(result=result)
        resp = client.post("/api/v1/ingest", json={"file_path": "doc.pdf"})
        assert resp.status_code == 200

    def test_post_ingest_success_response_shape(self, client):
        result = IngestionResult(document_id="sample_abc12345", sentences_count=42, status="success")
        app.dependency_overrides[get_ingestion_service] = lambda: self._make_mock_service(result=result)
        resp = client.post("/api/v1/ingest", json={"file_path": "doc.pdf"})
        body = resp.json()

        assert body["status"] == "success"
        assert body["sentences_indexed"] == 42
        assert "time_elapsed_seconds" in body
        assert isinstance(body["time_elapsed_seconds"], float)

    def test_post_ingest_details_is_list_on_success(self, client):
        result = IngestionResult(document_id="sample_abc12345", sentences_count=42, status="success")
        app.dependency_overrides[get_ingestion_service] = lambda: self._make_mock_service(result=result)
        resp = client.post("/api/v1/ingest", json={"file_path": "doc.pdf"})
        body = resp.json()

        assert isinstance(body["details"], list)
        assert len(body["details"]) == 1
        assert body["details"][0]["document_id"] == "sample_abc12345"
        assert body["details"][0]["sentences_count"] == 42

    def test_post_ingest_non_pdf_returns_400(self, client):
        app.dependency_overrides[get_ingestion_service] = lambda: MagicMock(spec=IngestionService)
        resp = client.post("/api/v1/ingest", json={"file_path": "document.txt"})
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    def test_post_ingest_document_load_error_returns_failure_body(self, client):
        error = DocumentLoadError(file_path="missing.pdf", reason="File does not exist")
        app.dependency_overrides[get_ingestion_service] = lambda: self._make_mock_service(side_effect=error)
        resp = client.post("/api/v1/ingest", json={"file_path": "missing.pdf"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failure"
        assert body["sentences_indexed"] == 0
        assert isinstance(body["details"], str)
        assert "missing.pdf" in body["details"]

    def test_post_ingest_ingestion_error_returns_failure_body(self, client):
        error = IngestionError(message="Vector store write failed")
        app.dependency_overrides[get_ingestion_service] = lambda: self._make_mock_service(side_effect=error)
        resp = client.post("/api/v1/ingest", json={"file_path": "doc.pdf"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failure"
        assert body["sentences_indexed"] == 0
        assert isinstance(body["details"], str)

    def test_post_ingest_details_is_string_on_failure(self, client):
        error = DocumentLoadError(file_path="x.pdf", reason="not found")
        app.dependency_overrides[get_ingestion_service] = lambda: self._make_mock_service(side_effect=error)
        resp = client.post("/api/v1/ingest", json={"file_path": "x.pdf"})

        assert isinstance(resp.json()["details"], str)

    def test_post_ingest_unexpected_error_returns_500(self, client):
        app.dependency_overrides[get_ingestion_service] = lambda: self._make_mock_service(
            side_effect=RuntimeError("boom")
        )
        resp = client.post("/api/v1/ingest", json={"file_path": "doc.pdf"})
        assert resp.status_code == 500

    def test_health_check_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


