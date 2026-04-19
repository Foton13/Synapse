"""Unit tests for the graph storage layer."""

from unittest.mock import MagicMock, patch

from src.graph_store import GraphStore


class TestGraphStore:
    """Tests for the GraphStore wrapper."""

    @patch("src.graph_store.GraphDatabase.driver")
    def test_init_creates_driver(self, mock_driver_init):
        GraphStore(uri="bolt://test", user="u", password="p")
        mock_driver_init.assert_called_once_with("bolt://test", auth=("u", "p"))

    @patch("src.graph_store.GraphDatabase.driver")
    def test_add_knowledge_runs_queries(self, mock_driver_init):
        mock_driver = MagicMock()
        mock_driver_init.return_value = mock_driver
        
        gs = GraphStore(uri="bolt://test", user="u", password="p")
        
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Data to add
        from src.processor import KnowledgeGraph, Relation
        kg = KnowledgeGraph(
            entities=["A", "B"],
            relations=[Relation(source="A", relation="LINKS", target="B")]
        )
        
        gs.add_knowledge(kg)
        
        # Verify two session.run calls
        assert mock_session.run.call_count == 2
        # Check first call (entities)
        first_call = mock_session.run.call_args_list[0]
        assert "UNWIND $entities" in first_call[0][0]
        # Check second call (rels)
        second_call = mock_session.run.call_args_list[1]
        assert "UNWIND $rels" in second_call[0][0]

    @patch("src.graph_store.GraphDatabase.driver")
    def test_query_graph_returns_tuples(self, mock_driver_init):
        mock_driver = MagicMock()
        mock_driver_init.return_value = mock_driver
        
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Mock result
        mock_result = MagicMock()
        mock_result.__iter__.return_value = [
            {"conn_name": "EntityB", "rel_type": "TYPE_X"}
        ]
        mock_session.run.return_value = mock_result
        
        gs = GraphStore()
        results = gs.query_graph("EntityA")
        
        assert results == [("EntityB", "TYPE_X")]
        mock_session.run.assert_called_once()

    @patch("src.graph_store.GraphDatabase.driver")
    def test_close_closes_driver(self, mock_driver_init):
        mock_driver = MagicMock()
        mock_driver_init.return_value = mock_driver
        
        gs = GraphStore()
        gs.close()
        mock_driver.close.assert_called_once()
