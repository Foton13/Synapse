"""Unit tests for the vector store module (ChromaDB wrapper)."""

import pytest

from src.vector_store import VectorStore, _split_text


class TestSplitText:
    """Tests for the internal _split_text helper."""

    def test_short_text_not_split(self):
        result = _split_text("Hello world", chunk_size=1000)
        assert result == ["Hello world"]

    def test_long_text_is_split(self):
        text = "Paragraph one.\n\n" * 50  # ~800 chars
        chunks = _split_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(c.strip() for c in chunks)

    def test_no_empty_chunks(self):
        text = "word " * 500
        chunks = _split_text(text, chunk_size=100, overlap=20)
        assert all(c for c in chunks)

    def test_single_line_long_text(self):
        text = "a" * 3000
        chunks = _split_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 3


class TestVectorStore:
    """Integration-style tests using a temporary ChromaDB instance."""

    @pytest.fixture()
    def store(self, tmp_path):
        """Create a VectorStore backed by a temp directory."""
        vs = VectorStore(persist_path=str(tmp_path / "chroma_test"))
        yield vs
        vs.close()

    def test_add_and_query(self, store):
        store.add_document(
            doc_id="doc1",
            text="Python is a programming language used for AI.",
            metadata={"filename": "python.md"},
        )
        results = store.query("programming language", n_results=1)
        assert len(results["ids"][0]) == 1
        # Short text → single chunk → id is doc_id unchanged
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

    def test_close_prevents_further_use(self, store):
        store.close()
        with pytest.raises(RuntimeError, match="closed"):
            store.add_document("x", "text", {})

    def test_chunked_document(self, store):
        """Long text should be split into multiple chunks."""
        long_text = ("This is a paragraph about AI.\n\n") * 100  # ~3 KB
        store.add_document("long", long_text, {"filename": "big.md"})

        results = store.query("artificial intelligence", n_results=5)
        # Should find at least one chunk
        assert len(results["ids"][0]) >= 1
        # Chunk ids should contain the base doc_id
        assert any("long" in id_ for id_ in results["ids"][0])
