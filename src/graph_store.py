"""
Synapse — Neo4j graph storage layer.

Manages the knowledge graph in Neo4j, including entity creation,
relationship storage, and graph queries.
"""

import logging
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.config import settings

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
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        logger.debug("Neo4j driver created for %s", self.uri)

    # --- Context Manager ---------------------------------------------------

    def __enter__(self) -> "GraphStore":
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

    def add_knowledge(self, kg_data: Any) -> None:
        """
        Persist a ``KnowledgeGraph`` into Neo4j.

        Creates ``Entity`` nodes via ``MERGE`` and ``RELATED`` edges
        between them. Relationship endpoints are merged (not matched)
        so that relations whose source/target don't appear in the
        entity list are still persisted rather than silently dropped.

        Args:
            kg_data: A ``KnowledgeGraph`` instance with ``.entities``
            and ``.relations``.
        """
        with self.driver.session() as session:
            session.run(
                "UNWIND $entities AS name "
                "MERGE (e:Entity {name: name})",
                entities=[e for e in kg_data.entities],
            )

            session.run(
                "UNWIND $rels AS rel "
                "MERGE (a:Entity {name: rel.source}) "
                "MERGE (b:Entity {name: rel.target}) "
                "MERGE (a)-[r:RELATED {type: rel.type}]->(b)",
                rels=[
                    {"source": rel.source, "target": rel.target, "type": rel.relation}
                    for rel in kg_data.relations
                ],
            )

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

