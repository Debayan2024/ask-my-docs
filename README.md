# 🔍 Knowledge Assistant — Enterprise RAG System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=flat&logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB%20%2F%20FAISS-FF6B6B?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**A production-grade Retrieval-Augmented Generation (RAG) system that answers questions grounded in your private document corpus — not the open internet.**

[Features](#-features) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [Usage](#-usage) · [Configuration](#%EF%B8%8F-configuration) · [API Reference](#-api-reference) · [Roadmap](#-roadmap)

</div>

---

## 📌 Overview

Knowledge Assistant is an end-to-end RAG pipeline that lets you ask natural language questions against **your own documents** (PDFs, DOCX, TXT, Markdown) and receive grounded, source-cited answers powered by LLMs.

Built for enterprise use cases — private documentation, policy libraries, technical knowledge bases — where accuracy and traceability matter more than generality.

> **Key principle:** The system answers from *your documents first*. When the answer isn't in your corpus, it says so — no hallucination.

---

## ✨ Features

- 📄 **Multi-format ingestion** — PDF, DOCX, TXT, Markdown via LangChain document loaders
- ✂️ **Intelligent chunking** — Configurable `RecursiveCharacterTextSplitter` with overlap for context continuity
- 🧠 **Pluggable embeddings** — Sentence Transformers (local), OpenAI, or Cohere embeddings
- 🗃️ **Swappable vector stores** — ChromaDB (default, persisted), FAISS, Pinecone
- 🤖 **Flexible LLM backends** — Ollama (Mistral / Llama 3, fully local), OpenAI GPT-4, Anthropic Claude
- 🌐 **REST API** — FastAPI with async endpoints, ready for integration
- 🐳 **Dockerized** — Single `docker-compose up` launches the API + Ollama service
- 📊 **Retrieval metrics** — Retrieval precision benchmarking and qualitative analysis utilities

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────┐
                        │         INGESTION PIPELINE       │
                        │                                  │
  Raw Documents ──────► │  Load → Chunk → Embed → Store   │
  (PDF/DOCX/TXT)        │                                  │
                        │  run_ingest.py                   │
                        └──────────────┬──────────────────┘
                                       │  persisted vectors
                                       ▼
                               ┌───────────────┐
                               │   Vector DB   │
                               │  (ChromaDB /  │
                               │    FAISS)     │
                               └───────┬───────┘
                                       │  similarity search (top-k)
                        ┌──────────────▼──────────────────┐
                        │         QUERY PIPELINE           │
                        │                                  │
  User Query ─────────► │  Embed Query                     │
                        │  → Retrieve Chunks               │
                        │  → Build Prompt                  │
                        │  → LLM (Ollama / GPT / Claude)   │
                        │  → Return Answer + Sources       │
                        │                                  │
                        │  app.py  (FastAPI)               │
                        └─────────────────────────────────┘
```

### Embedding Model
`all-MiniLM-L6-v2` (default) — 384-dimensional vectors, fast, runs fully offline via Sentence Transformers.

### Retrieval Strategy
Cosine similarity search over the vector store. Top-k chunks (configurable) are injected into the LLM prompt as grounding context.

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- Docker + Docker Compose
- [Ollama](https://ollama.com) (for local LLM inference)

### 1. Clone the repo

```bash
git clone https://github.com/Sumit1673/knowledge-assistant.git
cd knowledge-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama and pull the model

```bash
# Install Ollama (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# In terminal 1 — start the Ollama server
ollama serve

# In terminal 2 — pull Mistral (used by default)
ollama pull mistral
```

### 4. Configure the system

```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml to point to your documents directory and set your preferences
```

### 5. Ingest your documents

```bash
python run_ingest.py
```

This reads documents from the configured `source_dir`, chunks them, generates embeddings, and persists the vector store to `output/vector_store/`.

### 6. Start the API server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## 🐳 Docker Setup

Docker Compose runs two services:

| Service | Description |
|---------|-------------|
| `rag_api` | FastAPI application (port 8000) |
| `ollama` | Local LLM inference server (port 11434) |

> **Note:** Document ingestion (`run_ingest.py`) is intentionally separate from the API startup to keep cold-start time fast. Run it manually after adding new documents, or automate it via a file-upload trigger or cron job.

```bash
# Start all services
docker-compose up -d

# Ingest documents (run once, or after adding new docs)
docker-compose exec rag_api python run_ingest.py

# View logs
docker-compose logs -f rag_api

# Stop
docker-compose down
```

---

## 💬 Usage

### Via REST API

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

**Response:**
```json
{
  "answer": "The refund policy allows returns within 30 days of purchase...",
  "sources": [
    {"document": "policy_handbook.pdf", "page": 12, "score": 0.91}
  ]
}
```

### Via Python client

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "Summarize the onboarding process"}
)
print(response.json()["answer"])
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to customise the pipeline:

```yaml
# Document ingestion
ingestion:
  source_dir: "data/documents"       # Folder with your source documents
  chunk_size: 500                    # Characters per chunk
  chunk_overlap: 50                  # Overlap between consecutive chunks

# Embeddings
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"   # or "openai", "cohere"

# Vector store
vector_store:
  type: "chroma"                     # "chroma" | "faiss" | "pinecone"
  persist_dir: "output/vector_store"

# LLM
llm:
  provider: "ollama"                 # "ollama" | "openai" | "anthropic"
  model: "mistral"                   # Model name for the chosen provider
  temperature: 0.1

# Retrieval
retrieval:
  top_k: 4                           # Number of chunks to retrieve per query
```

---

## 📂 Project Structure

```
knowledge-assistant/
├── app.py                  # FastAPI application & query endpoint
├── run_ingest.py           # Document ingestion pipeline (run once / on update)
├── config/
│   └── config.yaml         # Runtime configuration
├── src/
│   ├── ingestion/          # Document loaders, text splitters
│   ├── embeddings/         # Embedding model wrappers
│   ├── vector_store/       # VectorStoreManager (Chroma / FAISS)
│   ├── llm/                # LLM handler (Ollama / OpenAI / Anthropic)
│   └── rag/                # RAGQueryHandler — retrieval + prompt + generation
├── data/
│   └── documents/          # ← Drop your source documents here
├── output/
│   └── vector_store/       # Auto-generated persisted vector store
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Ask a question against the document corpus |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

Full schema available at `/docs` when the server is running.

---

## 🔭 Roadmap

- [ ] Streaming responses via Server-Sent Events
- [ ] File upload endpoint that auto-triggers ingestion
- [ ] Hybrid search (dense + BM25 sparse retrieval)
- [ ] Multi-tenant document namespacing
- [ ] Evaluation harness (RAGAS metrics: faithfulness, context recall)
- [ ] HuggingFace Spaces demo with Groq backend

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 👤 Author

**Sumit Vaise** — Senior ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/sumit-vaise-6644514b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Sumit1673)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-6366F1?style=flat)](https://sumit1673.github.io)

---

<div align="center">
<sub>If this project was useful, consider giving it a ⭐</sub>
</div>
