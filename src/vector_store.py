"""
Synapse — ChromaDB vector storage layer.

Handles semantic embedding and retrieval of Markdown documents
using ChromaDB with sentence-transformers.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.utils import embedding_functions

from src.config import get_settings

logger = logging.getLogger("synapse")

__all__ = ["VectorStore"]

# Default chunk size / overlap for splitting large documents
_CHUNK_SIZE = 1_000
_CHUNK_OVERLAP = 200


def _split_text(text: str, chunk_size: int = _CHUNK_SIZE,
                overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks on paragraph / sentence boundaries.

    This is a lightweight splitter that avoids pulling in
    ``langchain-text-splitters`` as a hard dependency.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at the last double-newline (paragraph boundary)
        segment = text[start:end]
        break_pos = segment.rfind("\n\n")
        if break_pos == -1:
            # Fallback: break at last single newline
            break_pos = segment.rfind("\n")
        if break_pos > 0 and break_pos > chunk_size // 4:
            end = start + break_pos
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c]


class VectorStore:
    """
    Wrapper around ChromaDB for document embedding and semantic search.

    Supports context manager protocol for automatic cleanup::

        with VectorStore() as vs:
            vs.add_document(...)

    The persistent storage path is configurable via the ``CHROMA_DB_PATH``
    environment variable (default: ``./data/chromadb``).

    Large documents are automatically split into overlapping chunks
    before embedding so that each vector stays within the optimal
    context window of the embedding model.
    """

    def __init__(self, persist_path: str | None = None):
        path = persist_path or get_settings().chroma_db_path
        client = chromadb.PersistentClient(path=path)
        self._client: chromadb.PersistentClient | None = client
        self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self._collection: Collection | None = (
            client.get_or_create_collection(
                name="notes",
                embedding_function=self._embedding_fn,  # type: ignore[arg-type]
            )
        )
        logger.debug("ChromaDB collection 'notes' ready at %s", path)

    # --- Context Manager ---------------------------------------------------

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- Internal helpers ---------------------------------------------------

    def _ensure_open(self) -> Collection:
        """Return the active collection or raise if the store was closed."""
        if self._collection is None:
            raise RuntimeError("VectorStore is closed")
        return self._collection

    # --- Public API ---------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Add or update a document in the vector store.

        Long documents are automatically split into overlapping chunks
        and each chunk is stored as a separate vector with shared metadata.

        Args:
            doc_id:   Unique identifier (typically the file path).
            text:     Full text content of the document.
            metadata: Additional metadata (e.g. ``{"filename": "note.md"}``).
        """
        collection = self._ensure_open()
        chunks = _split_text(text)

        ids = [
            f"{doc_id}::chunk_{i}" if len(chunks) > 1 else doc_id
            for i in range(len(chunks))
        ]
        metas = [
            {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]

        collection.upsert(ids=ids, documents=chunks, metadatas=metas)
        logger.info("Indexed document: %s (%d chunk(s))", doc_id, len(chunks))

    def query(self, query_text: str, n_results: int = 3) -> dict[str, Any]:
        """
        Perform a semantic search against indexed documents.

        Args:
            query_text: Natural-language query.
            n_results:  Maximum number of results to return.

        Returns:
            ChromaDB query result dict with ``documents``, ``metadatas``,
            ``distances``, and ``ids`` keys.
        """
        collection = self._ensure_open()
        result = collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return dict(result)

    def close(self) -> None:
        """Gracefully release ChromaDB resources.

        After calling ``close()`` the store is **not** reusable.
        Any subsequent ``add_document`` / ``query`` call will raise
        ``RuntimeError``.
        """
        if self._collection is None:
            return
        self._collection = None
        # Ensure the PersistentClient file handles are released
        try:
            if self._client is not None:
                del self._client
        except Exception:  # noqa: BLE001
            pass
        self._client = None
        logger.debug("ChromaDB resources released")
