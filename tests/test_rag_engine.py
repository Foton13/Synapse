"""Unit tests for the RAG engine module."""

from unittest.mock import MagicMock, patch

from src.rag_engine import answer_question


class TestAnswerQuestion:
    """Tests for the answer_question function."""

    def test_answer_returns_string(self):
        """answer_question should always return a string."""
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [["Some doc text"]], "ids": [["d1"]]}

        mock_gs = MagicMock()
        mock_gs.query_graph.return_value = [("Neo4j", "uses")]

        mock_llm = MagicMock()

        # Mock the structured extraction helper
        mock_entity_obj = MagicMock()
        mock_entity_obj.name = "Neo4j"

        # Mock the final answer chain
        answer_chain = MagicMock()
        answer_msg = MagicMock()
        answer_msg.content = "Neo4j is used for graph storage."
        answer_chain.invoke.return_value = answer_msg

        with patch("src.rag_engine.extract_structured", return_value=mock_entity_obj), \
             patch("src.rag_engine.PromptTemplate.from_template") as mock_ft:
            
            mock_ft.return_value.__or__.return_value = answer_chain
            result = answer_question("What is Neo4j?", mock_vs, mock_gs, mock_llm)

        assert isinstance(result, str)
        assert "Neo4j" in result

    def test_answer_handles_entity_extraction_failure(self):
        """If entity extraction fails, the answer should still be generated."""
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [["doc text"]], "ids": [["d1"]]}

        mock_gs = MagicMock()
        mock_llm = MagicMock()

        # Final answer chain
        answer_chain = MagicMock()
        answer_msg = MagicMock()
        answer_msg.content = "Based on the context..."
        answer_chain.invoke.return_value = answer_msg

        with patch("src.rag_engine.extract_structured", side_effect=Exception("LLM failed")), \
             patch("src.rag_engine.PromptTemplate.from_template") as mock_ft:

            mock_ft.return_value.__or__.return_value = answer_chain
            result = answer_question("What?", mock_vs, mock_gs, mock_llm)

        assert isinstance(result, str)
        # Graph store should NOT have been called because extraction failed
        mock_gs.query_graph.assert_not_called()

    def test_answer_handles_empty_vector_results(self):
        """Should work even when no documents are found in vector store."""
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [], "ids": []}

        mock_gs = MagicMock()
        mock_gs.query_graph.return_value = []

        mock_llm = MagicMock()

        mock_entity_obj = MagicMock()
        mock_entity_obj.name = "Test"

        answer_chain = MagicMock()
        answer_msg = MagicMock()
        answer_msg.content = "I don't have enough context."
        answer_chain.invoke.return_value = answer_msg

        with patch("src.rag_engine.extract_structured", return_value=mock_entity_obj), \
             patch("src.rag_engine.PromptTemplate.from_template") as mock_ft:

            mock_ft.return_value.__or__.return_value = answer_chain
            result = answer_question("Unknown?", mock_vs, mock_gs, mock_llm)

        assert isinstance(result, str)
