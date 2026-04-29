"""
Synapse — Global configuration.

Uses pydantic-settings to safely load and validate environment variables.
Lazy initialization prevents crashes at import time when env vars are missing.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings", "reload_settings"]


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and/or .env file.
    It provides fail-fast validation at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    llm_provider: str = "ollama"
    ollama_model: str = "llama3"
    openai_model: str = "gpt-4o"
    openai_api_key: str | None = None

    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str  # Required, will fail fast if not provided

    # ChromaDB Settings
    chroma_db_path: str = "./data/chromadb"

    @field_validator("neo4j_password")
    @classmethod
    def password_not_weak(cls, v: str) -> str:
        """Ensure the Neo4j password is at least 8 characters long."""
        if len(v) < 8:
            raise ValueError(
                "NEO4J_PASSWORD must be at least 8 characters long"
            )
        return v

    @field_validator("llm_provider")
    @classmethod
    def provider_is_valid(cls, v: str) -> str:
        """Allow only known LLM providers."""
        allowed = {"ollama", "openai"}
        if v not in allowed:
            raise ValueError(
                f"LLM_PROVIDER must be one of {allowed}, got '{v}'"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Return the global settings instance (created lazily on first call)."""
    return Settings()  # type: ignore[call-arg]


def reload_settings() -> Settings:
    """Force re-read of settings (e.g. after .env changes at runtime)."""
    get_settings.cache_clear()
    return get_settings()
