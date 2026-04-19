"""
Synapse — CLI entry point.

Provides three commands:
- ``index``  — scan Markdown files and populate the knowledge base.
- ``query``  — look up entity connections in the knowledge graph.
- ``ask``    — ask a natural-language question answered via GraphRAG.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

from src.graph_store import GraphStore
from src.processor import ExtractionError, get_llm, process_note
from src.rag_engine import answer_question
from src.vector_store import VectorStore

app = typer.Typer(
    name="synapse",
    help="Synapse -- Transform your Markdown notes into an intelligent knowledge base.",
    add_completion=False,
)

logger = logging.getLogger("synapse")


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

    def _process_single_file(file: Path, vs: VectorStore, gs: GraphStore) -> int:
        typer.echo(f"  Processing {file.name} …")
        content = file.read_text(encoding="utf-8")

        # 1. Vector indexing
        vs.add_document(
            doc_id=str(file),
            text=content,
            metadata={"filename": file.name},
        )

        # 2. Graph indexing
        try:
            kg_data = process_note(content)
            gs.add_knowledge(kg_data)
            typer.echo(
                f" ✅ {len(kg_data.entities)} entities, "
                f"{len(kg_data.relations)} relations for {file.name}"
            )
            return 1
        except ExtractionError as e:
            typer.echo(f" ⚠️ Could not extract graph for {file.name}: {e}")
            return 0
        except Exception as e:
            logger.error("Error processing %s: %s", file.name, e)
            typer.echo(f"    ❌ Error for {file.name}: {e}")
            return 0

    with GraphStore() as graph_store, VectorStore() as vector_store:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(_process_single_file, f, vector_store, graph_store)
                for f in md_files
            ]
            for future in as_completed(futures):
                indexed += future.result()

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
        answer = answer_question(question, vector_store, graph_store, llm)
        typer.echo(f"\n🤖 AI: {answer}")


if __name__ == "__main__":
    app()
