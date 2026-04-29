"""
Synapse — RAG Engine.

Handles the core Retrieval-Augmented Generation logic combining
vector search, graph knowledge, and LLM orchestration.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.graph_store import GraphStore
from src.vector_store import VectorStore

logger = logging.getLogger("synapse")

__all__ = ["answer_question"]


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
    """
    # 1. Vector search — find semantically similar documents
    vector_results = vector_store.query(question)
    context_docs = vector_results["documents"][0] if vector_results["documents"] else []

    # 2. Graph search — extract the main entity, then look it up
    entity_parser = PydanticOutputParser(pydantic_object=_ExtractedEntity)

    entity_prompt = PromptTemplate(
        template=(
            "Extract the main entity from the following question.\n"
            "{format_instructions}\n\n"
            "Question: {question}"
        ),
        input_variables=["question"],
        partial_variables={
            "format_instructions": entity_parser.get_format_instructions(),
        },
    )
    entity_chain = entity_prompt | llm | entity_parser

    graph_results: list[tuple[str, str]] = []
    entity_name = ""
    try:
        entity_response = entity_chain.invoke({"question": question})
        entity_name = entity_response.name.strip()
        logger.debug("Extracted entity: %s", entity_name)
        graph_results = graph_store.query_graph(entity_name)
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)

    graph_context = "\n".join(
        f"{entity_name} -[{rel}]-> {conn}" for conn, rel in graph_results
    )

    # 3. Generate the final answer
    prompt = PromptTemplate.from_template(
        "You are a personal-notes assistant. Use the provided context to answer.\n\n"
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

    return (
        answer.content if hasattr(answer, "content") else str(answer)
    )
