"""Unit tests for the RAG engine module."""

from unittest.mock import MagicMock, patch

from src.rag_engine import answer_question


class TestAnswerQuestion:
    """Tests for the answer_question function."""

    def _make_mock_llm(self, entity_response: str, answer_response: str):
        """Build a mock LLM that returns different values per invocation."""
        llm = MagicMock()
        # The LLM is used inside LangChain chains (prompt | llm | parser),
        # so we mock at the chain level via side_effect.
        return llm

    def test_answer_returns_string(self):
        """answer_question should always return a string."""
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [["Some doc text"]], "ids": [["d1"]]}

        mock_gs = MagicMock()
        mock_gs.query_graph.return_value = [("Neo4j", "uses")]

        mock_llm = MagicMock()

        # Patch the chains created inside answer_question
        with patch("src.rag_engine.PromptTemplate") as mock_pt_cls, \
             patch("src.rag_engine.PydanticOutputParser") as mock_parser_cls:

            # Entity extraction chain
            mock_entity_obj = MagicMock()
            mock_entity_obj.name = "Neo4j"

            # Build chained mocks for pipe operator
            entity_chain = MagicMock()
            entity_chain.invoke.return_value = mock_entity_obj

            answer_chain = MagicMock()
            answer_msg = MagicMock()
            answer_msg.content = "Neo4j is used for graph storage."
            answer_chain.invoke.return_value = answer_msg

            # prompt | llm  => chain_step,  chain_step | parser => entity_chain
            mock_prompt_instance = MagicMock()
            chain_step = MagicMock()
            mock_prompt_instance.__or__ = MagicMock(return_value=chain_step)
            chain_step.__or__ = MagicMock(return_value=entity_chain)

            # Second prompt for the answer
            mock_answer_prompt = MagicMock()
            mock_answer_prompt.__or__ = MagicMock(return_value=answer_chain)

            mock_pt_cls.side_effect = [mock_prompt_instance, None]
            mock_pt_cls.from_template = MagicMock(return_value=mock_answer_prompt)

            result = answer_question("What is Neo4j?", mock_vs, mock_gs, mock_llm)

        assert isinstance(result, str)
        assert "Neo4j" in result

    def test_answer_handles_entity_extraction_failure(self):
        """If entity extraction fails, the answer should still be generated."""
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [["doc text"]], "ids": [["d1"]]}

        mock_gs = MagicMock()

        mock_llm = MagicMock()

        with patch("src.rag_engine.PromptTemplate") as mock_pt_cls, \
             patch("src.rag_engine.PydanticOutputParser"):

            # Entity chain raises
            entity_chain = MagicMock()
            entity_chain.invoke.side_effect = Exception("LLM failed")

            mock_prompt_instance = MagicMock()
            chain_step = MagicMock()
            mock_prompt_instance.__or__ = MagicMock(return_value=chain_step)
            chain_step.__or__ = MagicMock(return_value=entity_chain)

            # Answer chain works
            answer_chain = MagicMock()
            answer_msg = MagicMock()
            answer_msg.content = "Based on the context..."
            answer_chain.invoke.return_value = answer_msg

            mock_answer_prompt = MagicMock()
            mock_answer_prompt.__or__ = MagicMock(return_value=answer_chain)

            mock_pt_cls.side_effect = [mock_prompt_instance, None]
            mock_pt_cls.from_template = MagicMock(return_value=mock_answer_prompt)

            result = answer_question("What?", mock_vs, mock_gs, mock_llm)

        assert isinstance(result, str)
        # Graph store should NOT have been called
        mock_gs.query_graph.assert_not_called()

    def test_answer_handles_empty_vector_results(self):
        """Should work even when no documents are found in vector store."""
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [], "ids": []}

        mock_gs = MagicMock()
        mock_gs.query_graph.return_value = []

        mock_llm = MagicMock()

        with patch("src.rag_engine.PromptTemplate") as mock_pt_cls, \
             patch("src.rag_engine.PydanticOutputParser"):

            mock_entity_obj = MagicMock()
            mock_entity_obj.name = "Test"

            entity_chain = MagicMock()
            entity_chain.invoke.return_value = mock_entity_obj

            mock_prompt_instance = MagicMock()
            chain_step = MagicMock()
            mock_prompt_instance.__or__ = MagicMock(return_value=chain_step)
            chain_step.__or__ = MagicMock(return_value=entity_chain)

            answer_chain = MagicMock()
            answer_msg = MagicMock()
            answer_msg.content = "I don't have enough context."
            answer_chain.invoke.return_value = answer_msg

            mock_answer_prompt = MagicMock()
            mock_answer_prompt.__or__ = MagicMock(return_value=answer_chain)

            mock_pt_cls.side_effect = [mock_prompt_instance, None]
            mock_pt_cls.from_template = MagicMock(return_value=mock_answer_prompt)

            result = answer_question("Unknown?", mock_vs, mock_gs, mock_llm)

        assert isinstance(result, str)
