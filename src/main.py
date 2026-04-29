"""
Synapse — CLI entry point.

Provides three commands:
- ``index``  — scan Markdown files and populate the knowledge base.
- ``query``  — look up entity connections in the knowledge graph.
- ``ask``    — ask a natural-language question answered via GraphRAG.
"""

import logging
from pathlib import Path

import typer

from src.graph_store import GraphStore
from src.processor import ExtractionError, get_llm, process_note
from src.rag_engine import answer_question
from src.vector_store import VectorStore

__version__ = "0.1.0"

app = typer.Typer(
    name="synapse",
    help="Synapse — Transform your Markdown notes into an intelligent knowledge base.",
    add_completion=False,
)

logger = logging.getLogger("synapse")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"synapse {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Synapse — Transform your Markdown notes into an intelligent knowledge base."""


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def index(
    path: str = typer.Argument(..., help="Path to directory with Markdown files."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Scan Markdown files and index them into the vector DB and knowledge graph."""
    _setup_logging(verbose)

    root = Path(path)
    if not root.exists():
        typer.echo(f"❌ Path does not exist: {root}", err=True)
        raise typer.Exit(code=1)

    md_files = [
        f for f in root.rglob("*.md")
        if ".venv" not in f.parts and ".git" not in f.parts
    ]

    if not md_files:
        typer.echo("⚠️  No Markdown files found.")
        raise typer.Exit()

    typer.echo(f"📂 Found {len(md_files)} Markdown file(s)\n")

    indexed = 0

    with GraphStore() as graph_store, VectorStore() as vector_store:
        llm = get_llm()

        for file in md_files:
            typer.echo(f"  Processing {file.name} …")
            try:
                content = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                typer.echo(f"    ⚠️  Skipping {file.name}: not valid UTF-8")
                continue

            # 1. Vector indexing
            vector_store.add_document(
                doc_id=str(file),
                text=content,
                metadata={"filename": file.name},
            )

            # 2. Graph indexing
            try:
                kg_data = process_note(content, llm=llm)
                graph_store.add_knowledge(kg_data)
                typer.echo(
                    f"    ✅ {len(kg_data.entities)} entities, "
                    f"{len(kg_data.relations)} relations"
                )
                indexed += 1
            except ExtractionError as e:
                typer.echo(f"    ⚠️ Could not extract graph for {file.name}: {e}")
            except Exception as e:
                logger.error("Error processing %s: %s", file.name, e)
                typer.echo(f"    ❌ Error for {file.name}: {e}")

    typer.echo(f"\n✨ Done — {indexed}/{len(md_files)} files indexed successfully.")


@app.command()
def query(
    entity: str = typer.Argument(..., help="Entity name to look up."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Find all connections for a specific entity in the knowledge graph."""
    _setup_logging(verbose)

    with GraphStore() as graph_store:
        results = graph_store.query_graph(entity)

        if not results:
            typer.echo(f"No connections found for '{entity}'.")
            return

        typer.echo(f"🔗 Connections for '{entity}':\n")
        for connected, rel_type in results:
            typer.echo(f"  • {entity}  ─[{rel_type}]→  {connected}")


@app.command()
def ask(
    question: str = typer.Argument(
        ..., help="Natural-language question about your notes."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Ask an AI question about your notes using GraphRAG (vector + graph context)."""
    _setup_logging(verbose)

    with GraphStore() as graph_store, VectorStore() as vector_store:
        llm = get_llm()
        try:
            answer = answer_question(question, vector_store, graph_store, llm)
        except ValueError as e:
            typer.echo(f"❌ {e}", err=True)
            raise typer.Exit(code=1) from e
        typer.echo(f"\n🤖 AI: {answer}")


if __name__ == "__main__":
    app()
