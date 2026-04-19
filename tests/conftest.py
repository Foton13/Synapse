"""Shared pytest fixtures for Synapse tests."""

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch, tmp_path):
    """Ensure tests never hit real services by default."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chromadb"))
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:17687")  # unlikely port
    monkeypatch.setenv("NEO4J_USER", "test")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
