
import pytest

from src.vector_store import VectorStore


@pytest.fixture
def temp_chroma_path(tmp_path):
    path = tmp_path / "chromadb"
    yield str(path)
    # No explicit shutil.rmtree here to avoid PermissionError on Windows
    # pytest's tmp_path will be cleaned up by the system eventually

@pytest.fixture
def vector_store(temp_chroma_path):
    return VectorStore(persist_path=temp_chroma_path)

def test_add_and_query_documents(vector_store):
    vector_store.add_document(
        doc_id="test_doc",
        text="Synapse is an intelligent knowledge base",
        metadata={"filename": "test.md"}
    )
    
    results = vector_store.query("knowledge base")
    
    assert len(results["documents"][0]) > 0
    assert "Synapse is an intelligent knowledge base" in results["documents"][0]
