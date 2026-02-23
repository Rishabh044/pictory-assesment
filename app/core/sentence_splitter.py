"""Sentence splitter service for semantic search."""

import hashlib
from typing import List

import nltk
from langchain_core.documents import Document


class SentenceSplitter:
    """Splits text into sentences with stable IDs and metadata."""

    def __init__(self):
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    def _generate_content_hash(self, text: str) -> str:
        """Generate SHA256 hash of text, returning first 8 characters."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    def _generate_sentence_id(
        self, document_id: str, sentence_index: int, sentence_text: str
    ) -> str:
        """Generate stable sentence ID in format: {document_id}::{sentence_index}::{content_hash[:8]}"""
        content_hash = self._generate_content_hash(sentence_text)
        return f"{document_id}::{sentence_index}::{content_hash}"

    def split_text(self, text: str, document_id: str) -> List[Document]:
        """
        Split text into sentences with metadata.

        Args:
            text: The text to split into sentences.
            document_id: Identifier for the source document.

        Returns:
            List of LangChain Document objects, each containing:
            - page_content: The sentence text
            - metadata: dict with document_id, sentence_id, sentence_text,
                       sentence_index, start_offset, end_offset
        """
        if not text or not text.strip():
            return []

        sentences = nltk.sent_tokenize(text)

        if not sentences:
            return []

        documents = []
        search_start = 0

        for index, sentence in enumerate(sentences):
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue

            start_offset = text.find(sentence_stripped, search_start)
            if start_offset == -1:
                start_offset = search_start
            end_offset = start_offset + len(sentence_stripped)
            search_start = end_offset

            sentence_id = self._generate_sentence_id(
                document_id, index, sentence_stripped
            )

            metadata = {
                "document_id": document_id,
                "sentence_id": sentence_id,
                "sentence_text": sentence_stripped,
                "sentence_index": index,
                "start_offset": start_offset,
                "end_offset": end_offset,
            }

            documents.append(Document(page_content=sentence_stripped, metadata=metadata))

        return documents
