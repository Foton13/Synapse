"""Shared pytest fixtures for Synapse tests."""

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch, tmp_path):
    """Ensure tests never hit real services by default.

    We set env vars *and* clear the ``get_settings`` lru_cache so that
    each test gets a fresh ``Settings`` instance built from the patched env.
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chromadb"))
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:17687")  # unlikely port
    monkeypatch.setenv("NEO4J_USER", "test")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_secure_password")  # >=8 chars

    # Clear the cached settings so the next call re-reads the env
    from src.config import get_settings
    get_settings.cache_clear()

    yield

    # Clean up after test
    get_settings.cache_clear()
