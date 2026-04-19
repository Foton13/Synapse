"""Unit tests for the vector store module (ChromaDB wrapper)."""

import pytest

from src.vector_store import VectorStore


class TestVectorStore:
    """Integration-style tests using a temporary ChromaDB instance."""

    @pytest.fixture()
    def store(self, tmp_path):
        """Create a VectorStore backed by a temp directory."""
        return VectorStore(persist_path=str(tmp_path / "chroma_test"))

    def test_add_and_query(self, store):
        store.add_document(
            doc_id="doc1",
            text="Python is a programming language used for AI.",
            metadata={"filename": "python.md"},
        )
        results = store.query("programming language", n_results=1)
        assert len(results["ids"][0]) == 1
        assert results["ids"][0][0] == "doc1"

    def test_upsert_overwrites(self, store):
        store.add_document("doc1", "Old text", {"filename": "a.md"})
        store.add_document("doc1", "New text", {"filename": "a.md"})

        results = store.query("New text", n_results=1)
        assert results["documents"][0][0] == "New text"

    def test_query_empty_collection(self, store):
        results = store.query("anything", n_results=5)
        assert results["ids"][0] == []

    def test_multiple_documents(self, store):
        store.add_document("d1", "Machine learning and neural networks", 
                           {"filename": "ml.md"})
        store.add_document("d2", "Cooking recipes for pasta", {"filename": "cook.md"})
        store.add_document("d3", "Deep learning with PyTorch", {"filename": "dl.md"})

        results = store.query("artificial intelligence", n_results=2)
        returned_ids = set(results["ids"][0])
        # ML and DL docs should be more relevant than cooking
        assert "d2" not in returned_ids
