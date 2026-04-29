"""Unit tests for the configuration module."""

import pytest
from pydantic import ValidationError

from src.config import Settings


class TestSettings:
    """Tests for Settings validation."""

    def test_valid_settings(self):
        s = Settings(
            neo4j_password="secure_pass_123",
            llm_provider="ollama",
        )
        assert s.neo4j_password == "secure_pass_123"

    def test_short_password_rejected(self):
        with pytest.raises(ValidationError, match="at least 8 characters"):
            Settings(neo4j_password="short")

    def test_empty_password_rejected(self):
        with pytest.raises(ValidationError, match="at least 8 characters"):
            Settings(neo4j_password="")

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError, match="must be one of"):
            Settings(
                neo4j_password="secure_pass_123",
                llm_provider="anthropic",
            )

    def test_ollama_provider_accepted(self):
        s = Settings(neo4j_password="secure_pass_123", llm_provider="ollama")
        assert s.llm_provider == "ollama"

    def test_openai_provider_accepted(self):
        s = Settings(neo4j_password="secure_pass_123", llm_provider="openai")
        assert s.llm_provider == "openai"
