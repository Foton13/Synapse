"""Integration tests for GraphStore using testcontainers (requires Docker)."""

import subprocess

import pytest
from testcontainers.neo4j import Neo4jContainer

from src.graph_store import GraphStore


def docker_is_running() -> bool:
    """Check if Docker daemon is available using subprocess (no extra deps)."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not docker_is_running(), reason="Docker is not running"
)


@pytest.fixture(scope="module")
def neo4j_container():
    # Use a specific image version for consistency
    container = Neo4jContainer("neo4j:5-community")
    container.start()
    yield container
    container.stop()


@pytest.fixture()
def graph_store(neo4j_container):
    """Create a GraphStore per test and clean up data before each run."""
    gs = GraphStore(
        uri=neo4j_container.get_connection_url(),
        user="neo4j",
        password=neo4j_container.password,
    )
    # Clean up stale data from previous tests
    with gs.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield gs
    gs.close()


def test_add_and_query_knowledge(graph_store):
    from src.processor import KnowledgeGraph, Relation

    kg_data = KnowledgeGraph(
        entities=["Python", "Neo4j"],
        relations=[Relation(source="Python", relation="uses", target="Neo4j")],
    )

    graph_store.add_knowledge(kg_data)

    results = graph_store.query_graph("Python")
    assert ("Neo4j", "uses") in results


def test_verify_connection(graph_store):
    assert graph_store.verify_connection() is True


def test_context_manager(neo4j_container):
    with GraphStore(
        uri=neo4j_container.get_connection_url(),
        user="neo4j",
        password=neo4j_container.password,
    ) as gs:
        assert gs.verify_connection() is True
    # Once closed, verify_connection might fail, or driver is closed.
    # We can at least check it doesn't throw during the with block.
