"""
Synapse — Neo4j graph storage layer.

Manages the knowledge graph in Neo4j, including entity creation,
relationship storage, and graph queries.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.config import get_settings
from src.processor import KnowledgeGraph

logger = logging.getLogger("synapse")

__all__ = ["GraphStore"]


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
            self.uri, auth=(self.user, self.password)
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
        except Exception as exc:  # noqa: BLE001
            # Non-fatal: the DB may not support constraints (e.g. Community edition)
            logger.warning("Could not create Entity.name constraint: %s", exc)

    # --- Context Manager ---------------------------------------------------

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
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

    def add_knowledge(self, kg_data: KnowledgeGraph) -> None:
        """
        Persist a ``KnowledgeGraph`` into Neo4j **atomically**.

        Creates ``Entity`` nodes via ``MERGE`` and ``RELATED`` edges
        between them inside a single transaction so that a partial
        failure never leaves the graph in an inconsistent state.

        Args:
            kg_data: A ``KnowledgeGraph`` instance with ``.entities``
                     and ``.relations``.
        """
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                tx.run(
                    "UNWIND $entities AS name "
                    "MERGE (e:Entity {name: name})",
                    entities=kg_data.entities,
                )

                tx.run(
                    "UNWIND $rels AS rel "
                    "MERGE (a:Entity {name: rel.source}) "
                    "MERGE (b:Entity {name: rel.target}) "
                    "MERGE (a)-[r:RELATED {type: rel.type}]->(b)",
                    rels=[
                        {
                            "source": rel.source,
                            "target": rel.target,
                            "type": rel.relation,
                        }
                        for rel in kg_data.relations
                    ],
                )

                tx.commit()

        logger.info(
            "Stored %d entities, %d relations",
            len(kg_data.entities),
            len(kg_data.relations),
        )

    def query_graph(self, entity_name: str) -> list[tuple[str, str]]:
        """
        Find all connections for a given entity.

        Args:
            entity_name: Exact name of the entity to search.

        Returns:
            List of ``(connected_entity_name, relationship_type)`` tuples.
        """
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {name: $name})-[r]-(connected) "
                "RETURN connected.name as conn_name, r.type as rel_type",
                name=entity_name,
            )
            return [
                (record["conn_name"], record["rel_type"]) for record in result
            ]
