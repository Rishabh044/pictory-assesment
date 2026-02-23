"""Tests for the sentence splitter service."""

import pytest

from app.core.sentence_splitter import SentenceSplitter


@pytest.fixture
def splitter():
    """Create a SentenceSplitter instance."""
    return SentenceSplitter()


class TestSentenceSplitter:
    """Tests for SentenceSplitter class."""

    def test_split_multiple_sentences(self, splitter):
        """Split paragraph into sentences, verify count and content."""
        text = "The quick brown fox jumps over the lazy dog. It was a sunny day. The fox was happy."
        documents = splitter.split_text(text, "test_doc.txt")

        assert len(documents) == 3
        assert documents[0].page_content == "The quick brown fox jumps over the lazy dog."
        assert documents[1].page_content == "It was a sunny day."
        assert documents[2].page_content == "The fox was happy."

    def test_sentence_id_format(self, splitter):
        """Verify ID format: {doc_id}::{index}::{hash[:8]}."""
        text = "Hello world. This is a test."
        documents = splitter.split_text(text, "my_doc.txt")

        for doc in documents:
            sentence_id = doc.metadata["sentence_id"]
            parts = sentence_id.split("::")
            assert len(parts) == 3
            assert parts[0] == "my_doc.txt"
            assert parts[1].isdigit()
            assert len(parts[2]) == 8

    def test_sentence_id_stability(self, splitter):
        """Same input produces identical sentence IDs."""
        text = "The quick brown fox. Jumps over the lazy dog."

        documents1 = splitter.split_text(text, "doc.txt")
        documents2 = splitter.split_text(text, "doc.txt")

        assert len(documents1) == len(documents2)
        for doc1, doc2 in zip(documents1, documents2):
            assert doc1.metadata["sentence_id"] == doc2.metadata["sentence_id"]

    def test_character_offsets(self, splitter):
        """Verify start_offset and end_offset are correct."""
        text = "First sentence. Second sentence."
        documents = splitter.split_text(text, "test.txt")

        for doc in documents:
            start = doc.metadata["start_offset"]
            end = doc.metadata["end_offset"]
            assert text[start:end] == doc.page_content

    def test_empty_text(self, splitter):
        """Empty string returns empty list."""
        assert splitter.split_text("", "doc.txt") == []
        assert splitter.split_text("   ", "doc.txt") == []

    def test_single_sentence(self, splitter):
        """Single sentence text works correctly."""
        text = "This is a single sentence."
        documents = splitter.split_text(text, "single.txt")

        assert len(documents) == 1
        assert documents[0].page_content == "This is a single sentence."
        assert documents[0].metadata["sentence_index"] == 0
        assert documents[0].metadata["document_id"] == "single.txt"

    def test_metadata_completeness(self, splitter):
        """All required metadata fields present."""
        text = "Test sentence for metadata."
        documents = splitter.split_text(text, "meta_doc.txt")

        required_fields = [
            "document_id",
            "sentence_id",
            "sentence_text",
            "sentence_index",
            "start_offset",
            "end_offset",
        ]

        for doc in documents:
            for field in required_fields:
                assert field in doc.metadata, f"Missing field: {field}"

    def test_sentence_text_in_metadata_matches_content(self, splitter):
        """Verify sentence_text metadata matches page_content."""
        text = "First. Second. Third."
        documents = splitter.split_text(text, "doc.txt")

        for doc in documents:
            assert doc.metadata["sentence_text"] == doc.page_content

    def test_sentence_indices_sequential(self, splitter):
        """Sentence indices should be sequential starting from 0."""
        text = "One. Two. Three. Four."
        documents = splitter.split_text(text, "doc.txt")

        indices = [doc.metadata["sentence_index"] for doc in documents]
        assert indices == list(range(len(documents)))

    def test_different_document_ids_different_sentence_ids(self, splitter):
        """Same text with different doc IDs produces different sentence IDs."""
        text = "Test sentence."

        doc1 = splitter.split_text(text, "doc1.txt")[0]
        doc2 = splitter.split_text(text, "doc2.txt")[0]

        assert doc1.metadata["sentence_id"] != doc2.metadata["sentence_id"]
        assert doc1.metadata["sentence_id"].startswith("doc1.txt::")
        assert doc2.metadata["sentence_id"].startswith("doc2.txt::")
