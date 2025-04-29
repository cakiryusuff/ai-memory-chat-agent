# 🧠 Conversational Memory AI Agent

A modular, memory-enabled conversational assistant powered by [Pydantic AI Agents](https://github.com/pydantic/pydantic-ai), [ChromaDB](https://www.trychroma.com/), [PostgreSQL](https://www.postgresql.org/), and a local [Ollama](https://ollama.com/) LLM server. Designed to recall previous user interactions and generate contextually relevant responses using agent-based reasoning and vector similarity search.

---

## 🚀 Features

- ✨ **Context-Aware Conversations**: Remembers past user queries and reuses relevant information.
- 🧱 **Modular Agent Architecture**: Built with `pydantic_graph` for composable AI workflows.
- 🧠 **Long-Term Memory**: Uses ChromaDB (vector database) for embedding-based memory.
- 🗂️ **SQL Storage**: Stores raw chat logs in PostgreSQL for structured archiving.
- ⚙️ **Local LLM Support**: Runs with a locally hosted Ollama server (e.g. LLaMA 3, Qwen2).
- ✅ **Query Generation & Classification Agents**: Dynamically determines what prior knowledge is needed and filters embeddings accordingly.

---

## 🧰 Technologies Used

| Technology        | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **[Python 3.10+]**| Core programming language.                                                  |
| **[Pydantic AI](https://github.com/pydantic/pydantic-ai)** | Framework to create type-safe AI agents with OpenAI-like interfaces.         |
| **[pydantic-graph](https://github.com/pydantic/pydantic-ai/tree/main/libs/pydantic-graph)** | Allows defining and running directed graphs of AI agents. |
| **[Ollama](https://ollama.com/)** | Local LLM server (used here for `llama3.2`, `qwen2.5`, etc.). |
| **[ChromaDB](https://www.trychroma.com/)** | Vector store used for semantic memory recall.                                  |
| **[PostgreSQL](https://www.postgresql.org/)** | Relational database to persist structured chat logs.                           |
| **[psycopg2](https://pypi.org/project/psycopg2/)** | PostgreSQL driver for Python.                                                 |

---

## ⚙️ Setup Instructions

### 1. 🐍 Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 🛠️ Install & Run Ollama Locally
```bash
# Download Ollama (https://ollama.com/)
ollama run llama3
# or another model like
ollama run qwen2
```

Make sure Ollama runs at http://localhost:11434

### 3. 🧠 Start ChromaDB (Optional: persistent mode is already included)

No setup required; handled via chromadb.PersistentClient(path="vectordb").

4. 🐘 PostgreSQL Setup

```sql
CREATE DATABASE demodb;

CREATE TABLE chat_conversation (
    id SERIAL PRIMARY KEY,
    user_chat TEXT NOT NULL,
    gpt_chat TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Be sure to set your PostgreSQL password in your environment:

```bash
export DB_PASSWORD=your_password  # Or use a .env file
```

## 💬 Usage
```bash
python app.py
```

Interactive CLI commands:

/exit — Exit the chat

/list_all — List all saved conversations (from ChromaDB)

/del_last — Delete last conversation (from ChromaDB)

/del_all — Delete all saved conversations (from ChromaDB)