"""
Synapse — LLM-powered knowledge extraction from Markdown notes.

This module handles the extraction of entities and relationships
from text using Large Language Models (Ollama or OpenAI).
"""

import logging
import re
from typing import Any, TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import get_settings

logger = logging.getLogger("synapse")

__all__ = [
    "Relation",
    "KnowledgeGraph",
    "get_llm",
    "process_note",
    "extract_structured",
    "ExtractionError",
    "sanitize_entity_name",
]

# Default timeout (seconds) for LLM calls
LLM_TIMEOUT = 120

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
            raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
            timeout=LLM_TIMEOUT,
        )
    return ChatOllama(model=settings.ollama_model, timeout=LLM_TIMEOUT)


class ExtractionError(Exception):
    """Exception raised when knowledge extraction fails."""


_T = TypeVar("_T", bound=BaseModel)


def extract_structured(
    llm: BaseChatModel,
    pydantic_class: type[_T],
    template: str,
    input_variables: list[str],
    **invoke_kwargs: Any,
) -> _T:
    """Build a prompt → LLM → parser chain and invoke it.

    This is a shared helper that eliminates duplicated chain-building
    logic in *processor* and *rag_engine*.

    Args:
        llm: The language model to use.
        pydantic_class: Pydantic model class for structured output.
        template: Prompt template string (must contain ``{format_instructions}``).
        input_variables: Names of the user-supplied template variables.
        **invoke_kwargs: Values for the template variables.

    Returns:
        Parsed Pydantic model instance.
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    prompt = PromptTemplate(
        template=template,
        input_variables=input_variables,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    return cast(_T, chain.invoke(invoke_kwargs))


def sanitize_entity_name(name: str) -> str:
    """Normalize an entity name: strip whitespace and remove unsafe characters."""
    return re.sub(r"[^\w\s\-']", "", name.strip())


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
        # Truncate at the last newline before the limit to avoid mid-word cuts.
        # Guard: keep at least 100 chars to avoid degenerating to empty string.
        cut = content[:MAX_CONTENT_LENGTH].rfind("\n")
        content = content[:cut] if cut > 100 else content[:MAX_CONTENT_LENGTH]

    llm = llm or get_llm()

    try:
        result = extract_structured(
            llm=llm,
            pydantic_class=KnowledgeGraph,
            template=(
                "Analyze the following text and extract key entities "
                "and their relationships.\n"
                "{format_instructions}\n\n"
                "Text:\n{text}"
            ),
            input_variables=["text"],
            text=content,
        )

        # Sanitize all entity names coming from LLM output
        result.entities = [
            sanitize_entity_name(e) for e in result.entities if sanitize_entity_name(e)
        ]
        for rel in result.relations:
            rel.source = sanitize_entity_name(rel.source)
            rel.target = sanitize_entity_name(rel.target)

        return result
    except Exception as e:
        logger.error("Failed to extract knowledge graph: %s", e)
        raise ExtractionError(f"Failed to extract knowledge graph: {e}") from e
