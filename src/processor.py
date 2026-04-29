"""
Synapse — LLM-powered knowledge extraction from Markdown notes.

This module handles the extraction of entities and relationships
from text using Large Language Models (Ollama or OpenAI).
"""

import logging
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import get_settings

logger = logging.getLogger("synapse")

__all__ = ["Relation", "KnowledgeGraph", "get_llm", "process_note", "ExtractionError"]

# Maximum content length sent to LLM to avoid context window overflow / cost spikes
MAX_CONTENT_LENGTH = 50_000


class Relation(BaseModel):
    """Represents a directed relationship between two entities."""

    source: str = Field(description="The starting entity of the relationship")
    relation: str = Field(
        description="The type of relationship (e.g., 'related_to', "
        "'part_of', 'implemented_with')"
    )
    target: str = Field(description="The ending entity of the relationship")


class KnowledgeGraph(BaseModel):
    """Structured output from LLM extraction — entities and their relationships."""

    entities: list[str] = Field(description="List of unique entities found in the text")
    relations: list[Relation] = Field(
        description="List of relationships between entities"
    )


def get_llm() -> BaseChatModel:
    """
    Factory function that returns the configured LLM instance.

    Reads provider from application settings:
    - ``"openai"`` → ``ChatOpenAI(model="gpt-4o")``
    - anything else → ``ChatOllama`` with model from settings (default ``llama3``)

    Raises:
        ValueError: If ``LLM_PROVIDER=openai`` but ``OPENAI_API_KEY`` is not set.
    """
    settings = get_settings()
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when LLM_PROVIDER=openai"
            )
        return ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key)  # type: ignore[arg-type]
    return ChatOllama(model=settings.ollama_model)


class ExtractionError(Exception):
    """Exception raised when knowledge extraction fails."""


def process_note(
    content: str,
    llm: BaseChatModel | None = None,
) -> KnowledgeGraph:
    """
    Extract entities and relationships from a Markdown note using an LLM.

    Args:
        content: Raw text content of the Markdown note.
        llm: Optional pre-created LLM instance. If ``None``, one is
             created via ``get_llm()``.

    Returns:
        A ``KnowledgeGraph`` with extracted entities and relations.

    Raises:
        ExtractionError: If the content is empty or the extraction chain fails.
    """
    if not content.strip():
        raise ExtractionError("Empty content — nothing to extract")

    if len(content) > MAX_CONTENT_LENGTH:
        logger.warning(
            "Content truncated from %d to %d chars",
            len(content),
            MAX_CONTENT_LENGTH,
        )
        content = content[:MAX_CONTENT_LENGTH]

    llm = llm or get_llm()
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)

    prompt = PromptTemplate(
        template=(
            "Analyze the following text and extract key entities "
            "and their relationships.\n"
            "{format_instructions}\n\n"
            "Text:\n{text}"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"text": content})
        return cast(KnowledgeGraph, result)
    except Exception as e:
        logger.error("Failed to extract knowledge graph: %s", e)
        raise ExtractionError(f"Failed to extract knowledge graph: {e}") from e
