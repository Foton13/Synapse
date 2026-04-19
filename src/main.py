import typer
from pathlib import Path
from src.processor import process_note, get_llm
from src.graph_store import GraphStore
from src.vector_store import VectorStore
from langchain.prompts import PromptTemplate

app = typer.Typer()
graph_store = GraphStore()
vector_store = VectorStore()

@app.command()
def index(path: str):
    """
    Сканування Markdown файлів за вказаним шляхом та індексація в граф і векторну БД
    """
    root = Path(path)
    md_files = [f for f in root.rglob("*.md") if ".venv" not in f.parts and ".git" not in f.parts]
    
    typer.echo(f"Знайдено Markdown-файлів: {len(md_files)}")
    for file in md_files:
        try:
            typer.echo(f"Обробка {file.name}...")
            content = file.read_text(encoding='utf-8')
            
            # 1. Vector Indexing
            vector_store.add_document(
                doc_id=str(file),
                text=content,
                metadata={"filename": file.name}
            )
            
            # 2. Graph Indexing
            kg_data = process_note(content)
            if hasattr(kg_data, 'entities'):
                graph_store.add_knowledge(kg_data)
                typer.echo(f" - Збережено у граф: {len(kg_data.entities)} сутностей")
            else:
                typer.echo(f" - Помилка обробки графа для {file.name}: {kg_data}")
                
        except Exception as e:
            typer.echo(f" Помилка при обробці {file.name}: {e}")

@app.command()
def query(entity: str):
    """
    Пошук зв'язків для конкретної сутності в графі
    """
    results = graph_store.query_graph(entity)
    if not results:
        typer.echo(f"Зв'язків для '{entity}' не знайдено.")
        return
    
    typer.echo(f"Зв'язки для '{entity}':")
    for conn, rel in results:
        typer.echo(f" - {rel} -> {conn}")

@app.command()
def ask(question: str):
    """
    Запитати AI про ваші нотатки (GraphRAG)
    """
    # 1. Vector Search
    vector_results = vector_store.query(question)
    context_docs = vector_results['documents'][0] if vector_results['documents'] else []
    
    # 2. Graph Search (Entity extraction from question)
    llm = get_llm()
    entity_prompt = PromptTemplate.from_template(
        "Extract the main entity from the following question. Return only the entity name.\n\nQuestion: {question}"
    )
    entity_chain = entity_prompt | llm
    
    try:
        entity_response = entity_chain.invoke({"question": question})
        # Handle both ChatModel and LLM outputs
        entity_name = entity_response.content if hasattr(entity_response, 'content') else str(entity_response)
        entity_name = entity_name.strip()
        
        graph_results = graph_store.query_graph(entity_name)
    except Exception as e:
        typer.echo(f"Помилка при пошуку сутності: {e}")
        graph_results = []

    graph_context = "\n".join([f"{rel} -> {conn}" for conn, rel in graph_results])
    
    # 3. Generate Answer
    prompt = PromptTemplate.from_template(
        "Ви - помічник по особистим нотаткам. Використовуйте наданий контекст, щоб відповісти на питання.\n\n"
        "Векторний контекст:\n{vector_context}\n\n"
        "Графовий контекст (зв'язки):\n{graph_context}\n\n"
        "Питання: {question}\n"
        "Відповідь:"
    )
    
    chain = prompt | llm
    answer = chain.invoke({
        "vector_context": "\n".join(context_docs),
        "graph_context": graph_context,
        "question": question
    })
    
    final_answer = answer.content if hasattr(answer, 'content') else str(answer)
    typer.echo(f"\nAI: {final_answer}")

if __name__ == "__main__":
    try:
        app()
    finally:
        graph_store.close()

