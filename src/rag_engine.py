"""
Synapse — RAG Engine.

Handles the core Retrieval-Augmented Generation logic combining
vector search, graph knowledge, and LLM orchestration.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.graph_store import GraphStore
from src.processor import extract_structured, sanitize_entity_name
from src.vector_store import VectorStore

logger = logging.getLogger("synapse")

__all__ = ["answer_question"]

# Hard limit on question length to prevent abuse / accidental huge inputs
MAX_QUESTION_LENGTH = 1_000


class _ExtractedEntity(BaseModel):
    """Structured output for entity extraction from a question."""

    name: str = Field(description="The main entity name mentioned in the question")


def answer_question(
    question: str,
    vector_store: VectorStore,
    graph_store: GraphStore,
    llm: BaseChatModel,
) -> str:
    """
    Answers a natural language question using vector and graph context.

    Args:
        question: The user's query.
        vector_store: Instantiated vector store.
        graph_store: Instantiated graph store.
        llm: Instantiated LLM.

    Returns:
        The generated answer string.

    Raises:
        ValueError: If the question is empty or exceeds length limit.
    """
    # --- Input validation ---------------------------------------------------
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question too long ({len(question)} chars, "
            f"max {MAX_QUESTION_LENGTH})"
        )

    # 1. Vector search — find semantically similar documents
    vector_results = vector_store.query(question)
    docs = vector_results.get("documents") or []
    context_docs: list[str] = docs[0] if docs else []

    # 2. Graph search — extract the main entity, then look it up
    graph_results: list[tuple[str, str]] = []
    entity_name = ""
    try:
        entity_response = extract_structured(
            llm=llm,
            pydantic_class=_ExtractedEntity,
            template=(
                "Extract the main entity from the following question.\n"
                "{format_instructions}\n\n"
                "Question: {question}"
            ),
            input_variables=["question"],
            question=question,
        )
        entity_name = sanitize_entity_name(entity_response.name)
        logger.debug("Extracted entity: %s", entity_name)
        graph_results = graph_store.query_graph(entity_name)
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)

    graph_context = "\n".join(
        f"{entity_name} -[{rel}]-> {conn}" for conn, rel in graph_results
    )

    # 3. Generate the final answer
    prompt = PromptTemplate.from_template(
        "You are a personal-notes assistant. Use ONLY the provided context "
        "to answer. If the context does not contain enough information, "
        "say so.\n\n"
        "Vector context:\n{vector_context}\n\n"
        "Graph context (relationships):\n{graph_context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    chain = prompt | llm
    answer = chain.invoke({
        "vector_context": "\n".join(context_docs),
        "graph_context": graph_context,
        "question": question,
    })

    # Ensure we always return a plain string (content can be str | list)
    raw = answer.content if hasattr(answer, "content") else str(answer)
    return str(raw) if not isinstance(raw, str) else raw
