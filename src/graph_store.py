"""
Synapse — Neo4j graph storage layer.

Manages the knowledge graph in Neo4j, including entity creation,
relationship storage, and graph queries.
"""

from __future__ import annotations

import logging
import re
from types import TracebackType
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, DatabaseError, ServiceUnavailable

from src.config import get_settings
from src.processor import KnowledgeGraph, sanitize_entity_name

logger = logging.getLogger("synapse")

__all__ = ["GraphStore"]

# Maximum retry time for transient Neo4j errors (seconds)
_MAX_RETRY_TIME = 30.0


def _sanitize_relation_type(value: str) -> str:
    """Strip unsafe characters from a relationship type string."""
    return re.sub(r"[^\w\s\-]", "", value.strip())


class GraphStore:
    """
    Wrapper around the Neo4j driver for knowledge graph operations.

    Supports context manager protocol for automatic cleanup::

        with GraphStore() as gs:
            gs.add_knowledge(kg_data)
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_transaction_retry_time=_MAX_RETRY_TIME,
        )
        self._ensure_indexes()
        logger.debug("Neo4j driver created for %s", self.uri)

    # --- Indexes / Constraints ----------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create a uniqueness constraint on Entity.name (acts as index too)."""
        try:
            with self.driver.session() as session:
                session.run(
                    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                )
        except (ClientError, DatabaseError) as exc:
            # Non-fatal: the DB may not support constraints (Community edition)
            logger.warning("Could not create Entity.name constraint: %s", exc)
        except ServiceUnavailable as exc:
            logger.warning("Neo4j not reachable during index setup: %s", exc)

    # --- Context Manager ---------------------------------------------------

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- Public API ---------------------------------------------------------

    def verify_connection(self) -> bool:
        """Return ``True`` if Neo4j is reachable, ``False`` otherwise."""
        try:
            self.driver.verify_connectivity()
            return True
        except ServiceUnavailable:
            logger.warning("Neo4j is not reachable at %s", self.uri)
            return False

    def close(self) -> None:
        """Gracefully close the Neo4j driver."""
        self.driver.close()
        logger.debug("Neo4j driver closed")

    @staticmethod
    def _write_knowledge(
        tx: Any,
        entities: list[str],
        rels: list[dict[str, str]],
    ) -> None:
        """Transaction function executed inside ``session.execute_write``."""
        tx.run(
            "UNWIND $entities AS name "
            "MERGE (e:Entity {name: toLower(name)})"
            "SET e.display_name = name",
            entities=entities,
        )
        tx.run(
            "UNWIND $rels AS rel "
            "MERGE (a:Entity {name: toLower(rel.source)}) "
            "MERGE (b:Entity {name: toLower(rel.target)}) "
            "MERGE (a)-[r:RELATED {type: rel.type}]->(b)",
            rels=rels,
        )

    def add_knowledge(self, kg_data: KnowledgeGraph) -> None:
        """
        Persist a ``KnowledgeGraph`` into Neo4j **atomically**.

        All entity names and relation types are sanitized before storage
        to prevent injection of unexpected characters from LLM output.

        Creates ``Entity`` nodes via ``MERGE`` and ``RELATED`` edges
        between them inside a single transaction so that a partial
        failure never leaves the graph in an inconsistent state.

        Args:
            kg_data: A ``KnowledgeGraph`` instance with ``.entities``
                     and ``.relations``.
        """
        clean_entities = [
            sanitize_entity_name(e)
            for e in kg_data.entities
            if sanitize_entity_name(e)  # drop empty after sanitize
        ]

        rels = [
            {
                "source": sanitize_entity_name(rel.source),
                "target": sanitize_entity_name(rel.target),
                "type": _sanitize_relation_type(rel.relation),
            }
            for rel in kg_data.relations
            if sanitize_entity_name(rel.source)
            and sanitize_entity_name(rel.target)
        ]

        with self.driver.session() as session:
            session.execute_write(
                self._write_knowledge,
                clean_entities,
                rels,
            )

        logger.info(
            "Stored %d entities, %d relations",
            len(clean_entities),
            len(rels),
        )

    def query_graph(self, entity_name: str) -> list[tuple[str, str]]:
        """
        Find all connections for a given entity (case-insensitive).

        Args:
            entity_name: Name of the entity to search.

        Returns:
            List of ``(connected_entity_name, relationship_type)`` tuples.
        """
        clean_name = sanitize_entity_name(entity_name)
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity)-[r]-(connected) "
                "WHERE e.name = toLower($name) "
                "RETURN coalesce(connected.display_name, connected.name) "
                "       AS conn_name, r.type AS rel_type",
                name=clean_name,
            )
            return [
                (record["conn_name"], record["rel_type"]) for record in result
            ]
