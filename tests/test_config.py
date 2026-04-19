"""Unit tests for the configuration module."""

import pytest
from pydantic import ValidationError
from src.config import Settings

class TestConfig:
    """Tests for the Settings Pydantic model."""

    def test_settings_validation_fails_without_password(self, monkeypatch):
        # Clear env vars that might interfere
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        
        with pytest.raises(ValidationError):
            # Settings() without env vars will fail because neo4j_password is required
            Settings(_env_file=None) 

    def test_settings_load_from_env(self, monkeypatch):
        monkeypatch.setenv("NEO4J_PASSWORD", "secret123")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        
        settings = Settings(_env_file=None)
        assert settings.neo4j_password == "secret123"
        assert settings.llm_provider == "openai"

    def test_settings_defaults(self, monkeypatch):
        monkeypatch.setenv("NEO4J_PASSWORD", "test")
        
        settings = Settings(_env_file=None)
        assert settings.llm_provider == "ollama"
        assert settings.neo4j_user == "test"
        assert settings.neo4j_uri == "bolt://localhost:17687"
