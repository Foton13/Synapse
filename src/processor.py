"""
Synapse — LLM-powered knowledge extraction from Markdown notes.

This module handles the extraction of entities and relationships
from text using Large Language Models (Ollama or OpenAI).
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


from src.config import settings

logger = logging.getLogger("synapse")

__all__ = ["Relation", "KnowledgeGraph", "get_llm", "process_note", "ExtractionError"]


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
    """
    if settings.llm_provider == "openai":
        api_key = settings.openai_api_key or ""
        return ChatOpenAI(model="gpt-4o", api_key=api_key)  # type: ignore[arg-type]
    return ChatOllama(model=settings.ollama_model)

class ExtractionError(Exception):
    """Exception raised when knowledge extraction fails."""
    pass


def process_note(content: str) -> KnowledgeGraph:
    """
    Extract entities and relationships from a Markdown note using an LLM.

    Args:
        content: Raw text content of the Markdown note.

    Returns:
        A ``KnowledgeGraph`` with extracted entities and relations.

    Raises:
        ExtractionError: If the extraction chain fails.
    """
    llm = get_llm()
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
        from typing import cast
        result = chain.invoke({"text": content})
        return cast(KnowledgeGraph, result)
    except Exception as e:
        logger.error("Failed to extract knowledge graph: %s", e)
        raise ExtractionError(f"Failed to extract knowledge graph: {e}") from e

