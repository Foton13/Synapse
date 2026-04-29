"""Unit tests for the processor module (Pydantic models, LLM factory & sanitization)."""

from unittest.mock import MagicMock, patch

import pytest

from src.processor import (
    ExtractionError,
    KnowledgeGraph,
    Relation,
    get_llm,
    process_note,
    sanitize_entity_name,
)


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
        from src.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "openai_api_key", "sk-test")

        llm = get_llm()
        assert "openai" in type(llm).__name__.lower()

    def test_get_llm_defaults_to_ollama(self, monkeypatch):
        from src.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "llm_provider", "ollama")
        monkeypatch.setattr(settings, "ollama_model", "llama3")

        llm = get_llm()
        assert "ollama" in type(llm).__name__.lower()


class TestSanitizeEntityName:
    """Tests for the sanitize_entity_name utility."""

    @pytest.mark.parametrize("raw, expected", [
        ("  Python  ", "Python"),
        ("Neo4j!!!", "Neo4j"),
        ("O'Reilly", "O'Reilly"),
        ("normal_name", "normal_name"),
        ("with-dash", "with-dash"),
        ("  ", ""),
        ("", ""),
        ("hello<<<world>>>", "helloworld"),
        ("café", "café"),                      # unicode letters preserved
        ("Project (Alpha)", "Project Alpha"),   # parens removed
        ('@#$%^&*', ""),                        # only specials → empty
    ])
    def test_sanitize_various_inputs(self, raw: str, expected: str):
        assert sanitize_entity_name(raw) == expected


class TestProcessNote:
    """Tests for the process_note extraction function."""

    def test_process_note_empty_raises(self):
        with pytest.raises(ExtractionError, match="Empty content"):
            process_note("   ")

    def test_process_note_success(self):
        mock_kg = KnowledgeGraph(
            entities=["Python!!!"],
            relations=[Relation(source="A<<", relation="uses", target="B>>")],
        )

        with patch("src.processor.extract_structured", return_value=mock_kg):
            result = process_note("some text", llm=MagicMock())

        # entities should be sanitized
        assert result.entities == ["Python"]
        assert result.relations[0].source == "A"
        assert result.relations[0].target == "B"

    def test_process_note_failure_raises_extraction_error(self):
        with patch(
            "src.processor.extract_structured",
            side_effect=Exception("LLM Error"),
        ):
            with pytest.raises(ExtractionError, match="Failed to extract"):
                process_note("some text", llm=MagicMock())

    def test_process_note_drops_empty_entities_after_sanitize(self):
        mock_kg = KnowledgeGraph(
            entities=["Valid", "@#$"],  # second one becomes empty
            relations=[],
        )
        with patch("src.processor.extract_structured", return_value=mock_kg):
            result = process_note("some text", llm=MagicMock())
        assert result.entities == ["Valid"]
