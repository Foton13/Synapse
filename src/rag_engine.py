"""
Synapse — RAG Engine.

Handles the core Retrieval-Augmented Generation logic combining
vector search, graph knowledge, and LLM orchestration.
"""

import logging
from typing import Any

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger("synapse")


def answer_question(
    question: str,
    vector_store: Any,
    graph_store: Any,
    llm: Any,
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

    # 2. Graph search — extract the main entity from the question, then look it up
    entity_prompt = PromptTemplate.from_template(
        "Extract the main entity from the following question. "
        "Return only the entity name.\n\n"
        "Question: {question}"
    )
    entity_chain = entity_prompt | llm

    graph_results: list = []
    try:
        entity_response = entity_chain.invoke({"question": question})
        entity_name = (
            entity_response.content
            if hasattr(entity_response, "content")
            else str(entity_response)
        ).strip()
        logger.debug("Extracted entity: %s", entity_name)
        graph_results = graph_store.query_graph(entity_name)
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)

    graph_context = "\n".join(f"{rel} → {conn}" for conn, rel in graph_results)

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
