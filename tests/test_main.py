"""Unit tests for the CLI entry point (Typer commands)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.main import app

runner = CliRunner()


class TestIndexCommand:
    """Tests for the ``index`` CLI command."""

    def test_index_nonexistent_path(self):
        result = runner.invoke(app, ["index", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_index_empty_directory(self, tmp_path):
        result = runner.invoke(app, ["index", str(tmp_path)])
        assert "No Markdown files found" in result.output

    def test_index_processes_files(self, tmp_path):
        # Create a sample .md file
        md = tmp_path / "note.md"
        md.write_text("# Test\nSome content", encoding="utf-8")

        mock_kg = MagicMock()
        mock_kg.entities = ["Test"]
        mock_kg.relations = []

        with patch("src.main.GraphStore") as mock_gs_cls, \
             patch("src.main.VectorStore") as mock_vs_cls, \
             patch("src.main.process_note", return_value=mock_kg), \
             patch("src.main.get_llm"):

            mock_gs = MagicMock()
            mock_gs.__enter__ = MagicMock(return_value=mock_gs)
            mock_gs.__exit__ = MagicMock(return_value=False)
            mock_gs_cls.return_value = mock_gs

            mock_vs = MagicMock()
            mock_vs.__enter__ = MagicMock(return_value=mock_vs)
            mock_vs.__exit__ = MagicMock(return_value=False)
            mock_vs_cls.return_value = mock_vs

            result = runner.invoke(app, ["index", str(tmp_path)])

        assert result.exit_code == 0
        assert "1/1 files indexed" in result.output


class TestQueryCommand:
    """Tests for the ``query`` CLI command."""

    def test_query_no_results(self):
        with patch("src.main.GraphStore") as mock_gs_cls:
            mock_gs = MagicMock()
            mock_gs.__enter__ = MagicMock(return_value=mock_gs)
            mock_gs.__exit__ = MagicMock(return_value=False)
            mock_gs.query_graph.return_value = []
            mock_gs_cls.return_value = mock_gs

            result = runner.invoke(app, ["query", "FakeEntity"])

        assert "No connections found" in result.output

    def test_query_with_results(self):
        with patch("src.main.GraphStore") as mock_gs_cls:
            mock_gs = MagicMock()
            mock_gs.__enter__ = MagicMock(return_value=mock_gs)
            mock_gs.__exit__ = MagicMock(return_value=False)
            mock_gs.query_graph.return_value = [("Neo4j", "uses")]
            mock_gs_cls.return_value = mock_gs

            result = runner.invoke(app, ["query", "Python"])

        assert "Python" in result.output
        assert "Neo4j" in result.output
        assert "uses" in result.output


class TestAskCommand:
    """Tests for the ``ask`` CLI command."""

    def test_ask_returns_answer(self):
        with patch("src.main.GraphStore") as mock_gs_cls, \
             patch("src.main.VectorStore") as mock_vs_cls, \
             patch("src.main.get_llm"), \
             patch("src.main.answer_question", return_value="Test answer"):

            mock_gs = MagicMock()
            mock_gs.__enter__ = MagicMock(return_value=mock_gs)
            mock_gs.__exit__ = MagicMock(return_value=False)
            mock_gs_cls.return_value = mock_gs

            mock_vs = MagicMock()
            mock_vs.__enter__ = MagicMock(return_value=mock_vs)
            mock_vs.__exit__ = MagicMock(return_value=False)
            mock_vs_cls.return_value = mock_vs

            result = runner.invoke(app, ["ask", "What is Neo4j?"])

        assert result.exit_code == 0
        assert "Test answer" in result.output
