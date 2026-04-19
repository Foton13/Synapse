# Note-Graph-RAG 🧠

Цей інструмент перетворює ваші локальні Markdown-нотатки на інтелектуальну базу знань, поєднуючи **Векторний пошук** та **Графові зв'язки** (GraphRAG).

## 🚀 Особливості
- **LLM Extraction:** Автоматичне витягування сутностей та зв'язків з тексту.
- **Knowledge Graph:** Зберігання структури знань у Neo4j.
- **Semantic Search:** Швидкий пошук схожих ідей за допомогою ChromaDB.
- **Hybrid RAG:** Відповіді AI, що базуються як на семантиці, так і на структурі графа.

## 🛠 Технологічний стек
- **Python 3.11+**
- **LangChain / LlamaIndex** — Оркестрація LLM.
- **Neo4j** — Графова база даних.
- **ChromaDB** — Векторна база даних.
- **Typer** — CLI інтерфейс.
- **Ollama / OpenAI** — Моделі LLM.

## 📦 Встановлення та запуск

### 1. Клонування та середовище
```bash
git clone <repo-url>
cd note-graph-rag
python -m venv .venv
source .venv/bin/activate  # Unix
# або .venv\\Scripts\\activate на Windows
pip install -r requirements.txt
```

### 2. Запуск інфраструктури (Neo4j)
```bash
docker-compose up -d
```

### 3. Конфігурація
Створіть файл `.env` на основі прикладу (вказати провайдера LLM та пароль Neo4j).

### 4. Використання
- **Індексація нотаток:**
  ```bash
  python -m src.main index ./my_notes
  ```
- **Пошук зв'язків сутності:**
  ```bash
  python -m src.main query "Проект X"
  ```
- **Запитання до бази знань:**
  ```bash
  python -m src.main ask "Які технології використовуються в Проекті X?"
  ```

## 📐 Архітектура
`Markdown Files` $\rightarrow$ `LLM (Extraction)` $\rightarrow$ `Neo4j (Nodes/Edges)` $\&$ `ChromaDB (Embeddings)` $\rightarrow$ `RAG Query` $\rightarrow$ `AI Answer`
