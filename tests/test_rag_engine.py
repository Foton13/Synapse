from dataclasses import dataclass
import pytest
from src.rag_engine import answer_question

@dataclass
class MockResponse:
    content: str

class MockLLM:
    """A simple callable mock that LangChain can wrap in a RunnableLambda."""
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, input_data, **kwargs):
        self.calls.append(input_data)
        if not self.responses:
            return MockResponse(content="No more responses")
        return MockResponse(content=self.responses.pop(0))

    def invoke(self, input_data, **kwargs):
        """Invoke is called by RunnableSequence."""
        return self.__call__(input_data)

class TestRagEngine:
    """Tests for the RAG engine logic."""

    def test_answer_question_full_flow(self, monkeypatch):
        """Test the normal flow of answer generation."""
        from unittest.mock import MagicMock
        
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [["Document context about Python."]]}

        mock_gs = MagicMock()
        mock_gs.query_graph.return_value = [("Neo4j", "integrates_with")]

        # Entity extraction + Final answer
        llm = MockLLM(responses=["Python", "Python integrates with Neo4j."])

        question = "How does Python connect to databases?"
        answer = answer_question(question, mock_vs, mock_gs, llm)

        assert str(answer) == "Python integrates with Neo4j."
        mock_vs.query.assert_called_once_with(question)
        mock_gs.query_graph.assert_called_once_with("Python")
        assert len(llm.calls) == 2

    def test_answer_question_no_vector_results(self):
        """Test flow when vector search returns no documents."""
        from unittest.mock import MagicMock
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": []}

        mock_gs = MagicMock()
        mock_gs.query_graph.return_value = []

        llm = MockLLM(responses=["No entity", "I don't know."])

        answer = answer_question("Unknown topic?", mock_vs, mock_gs, llm)

        assert str(answer) == "I don't know."

    def test_answer_question_entity_extraction_failure(self):
        """Test flow when entity extraction fails (should still proceed with vector context)."""
        from unittest.mock import MagicMock
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"documents": [["Some context."]]}
        mock_gs = MagicMock()
        
        # A mock that fails on first call
        class FailingLLM(MockLLM):
            def __call__(self, input_data, **kwargs):
                if len(self.calls) == 0:
                    self.calls.append(input_data)
                    raise Exception("LLM Error")
                return super().__call__(input_data)

        llm = FailingLLM(responses=["Fallback answer."])

        answer = answer_question("Broken question?", mock_vs, mock_gs, llm)

        assert str(answer) == "Fallback answer."
        mock_gs.query_graph.assert_not_called()
        assert len(llm.calls) == 2
