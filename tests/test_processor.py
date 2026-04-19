"""Unit tests for the processor module (Pydantic models & LLM factory)."""

import pytest
from unittest.mock import MagicMock, patch

from src.processor import ExtractionError, KnowledgeGraph, Relation, get_llm, process_note


class TestRelationModel:
    """Tests for the Relation Pydantic model."""

    def test_create_valid_relation(self):
        rel = Relation(source="Python", relation="used_in", target="Synapse")
        assert rel.source == "Python"
        assert rel.relation == "used_in"
        assert rel.target == "Synapse"

    def test_relation_json_roundtrip(self):
        rel = Relation(source="A", relation="related_to", target="B")
        data = rel.model_dump()
        restored = Relation(**data)
        assert restored == rel

    def test_relation_missing_field_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Relation(source="A", relation="related_to")  # missing target


class TestKnowledgeGraphModel:
    """Tests for the KnowledgeGraph Pydantic model."""

    def test_create_empty_graph(self):
        kg = KnowledgeGraph(entities=[], relations=[])
        assert kg.entities == []
        assert kg.relations == []

    def test_create_populated_graph(self):
        kg = KnowledgeGraph(
            entities=["Python", "Neo4j", "ChromaDB"],
            relations=[
                Relation(source="Python", relation="integrates_with", 
                         target="Neo4j"),
                Relation(source="Python", relation="integrates_with", 
                         target="ChromaDB"),
            ],
        )
        assert len(kg.entities) == 3
        assert len(kg.relations) == 2


class TestGetLlm:
    """Tests for the LLM factory function."""

    def test_get_llm_respects_settings(self, monkeypatch):
        # We patch the settings object directly since it's already loaded
        from src.processor import settings
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")
        
        llm = get_llm()
        assert "openai" in type(llm).__name__.lower()

    def test_get_llm_defaults_to_ollama(self, monkeypatch):
        from src.processor import settings
        monkeypatch.setattr(settings, "llm_provider", "ollama")
        monkeypatch.setattr(settings, "ollama_model", "llama3")
        
        llm = get_llm()
        assert "ollama" in type(llm).__name__.lower()


class TestProcessNote:
    """Tests for the process_note extraction function."""

    def test_process_note_success(self):
        mock_kg = KnowledgeGraph(entities=["A"], relations=[])
        
        with patch("src.processor.get_llm"), \
             patch("src.processor.PromptTemplate") as mock_prompt_class, \
             patch("src.processor.PydanticOutputParser") as mock_parser_class:
            
            mock_prompt = mock_prompt_class.return_value
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_kg
            
            # Mock the pipe chain: prompt | llm | parser
            mock_prompt.__or__.return_value = MagicMock()
            mock_prompt.__or__.return_value.__or__.return_value = mock_chain
            
            result = process_note("some text")
            assert result == mock_kg

    def test_process_note_failure_raises_extraction_error(self):
        with patch("src.processor.get_llm"), \
             patch("src.processor.PromptTemplate") as mock_prompt_class, \
             patch("src.processor.PydanticOutputParser") as mock_parser_class:
            
            mock_prompt = mock_prompt_class.return_value
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("LLM Error")
            
            mock_prompt.__or__.return_value = MagicMock()
            mock_prompt.__or__.return_value.__or__.return_value = mock_chain
            
            with pytest.raises(ExtractionError) as excinfo:
                process_note("some text")
            
            assert "Failed to extract knowledge graph" in str(excinfo.value)
