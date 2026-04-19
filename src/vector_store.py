"""
Synapse — ChromaDB vector storage layer.

Handles semantic embedding and retrieval of Markdown documents
using ChromaDB with sentence-transformers.
"""

import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from src.config import settings

logger = logging.getLogger("synapse")

__all__ = ["VectorStore"]


class VectorStore:
    """
    Wrapper around ChromaDB for document embedding and semantic search.

    Supports context manager protocol for automatic cleanup::

        with VectorStore() as vs:
            vs.add_document(...)

    The persistent storage path is configurable via the ``CHROMA_DB_PATH``
    environment variable (default: ``./data/chromadb``).
    """

    def __init__(self, persist_path: str | None = None):
        path = persist_path or settings.chroma_db_path
        self.client: Any = chromadb.PersistentClient(path=path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection: Any = (
            self.client.get_or_create_collection(
                name="notes",
                embedding_function=self.embedding_fn,
            )
        )
        logger.debug("ChromaDB collection 'notes' ready at %s", path)

    # --- Context Manager ---------------------------------------------------

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self.close()

    # --- Public API ---------------------------------------------------------

    def add_document(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        """
        Add or update a document in the vector store.

        Args:
            doc_id:   Unique identifier (typically the file path).
            text:     Full text content of the document.
            metadata: Additional metadata (e.g. ``{"filename": "note.md"}``).
        """
        if self.collection is None:
            raise RuntimeError("VectorStore is closed")
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata],
        )
        logger.info("Indexed document: %s", doc_id)

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
        if self.collection is None:
            raise RuntimeError("VectorStore is closed")
        result = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return dict(result)

    def close(self) -> None:
        """Gracefully release ChromaDB resources."""
        if self.collection is None:
            return
        self.collection = None
        self.client = None
        logger.debug("ChromaDB resources released")


