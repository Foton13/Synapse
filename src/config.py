"""
Synapse — Global configuration.

Uses pydantic-settings to safely load and validate environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    openai_api_key: str | None = None

    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str  # Required, will fail fast if not provided

    # ChromaDB Settings
    chroma_db_path: str = "./data/chromadb"


# Global singleton settings instance
settings = Settings()  # type: ignore[call-arg]
