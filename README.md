<div align="center">

# 📚 DocuMind

**Chat with your documents. Get cited answers. Keep everything private.**

A production-grade RAG (Retrieval-Augmented Generation) chatbot built with LangChain, OpenAI, ChromaDB, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o-412991?style=flat-square&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B6B?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

[Features](#-features) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [Deploy](#-deploy)

</div>

---

## 🎯 What it does

Upload PDFs, text, markdown, or Word documents → ask natural-language questions → get answers **grounded in your documents** with inline citations.

Perfect for:
- 📑 Legal/compliance teams searching contracts
- 🧑‍💼 Consultants querying client reports
- 🏢 Internal knowledge bases (HR handbooks, SOPs, onboarding docs)
- 🎓 Researchers chatting with paper collections
- 🛠️ Engineering teams searching technical documentation

---

## ✨ Features

- **Multi-format ingestion** — PDF, TXT, Markdown, DOCX
- **Streaming responses** — tokens render in real time, not a spinner
- **Inline citations** — every answer points to the source file and page
- **Source inspector** — expand to see the exact chunks the LLM used
- **Per-file management** — add or remove individual documents; clear the whole base
- **Model switching** — toggle between `gpt-4o-mini` (fast, cheap) and `gpt-4o` (smart)
- **Persistent vector store** — your knowledge base survives restarts
- **Clean codebase** — typed Python, LCEL chains, small focused modules
- **Docker-ready** — one-line deploy to Render, Railway, Fly, or any container host

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+ (3.12 recommended)
- An OpenAI API key → [platform.openai.com](https://platform.openai.com/api-keys)
- [`uv`](https://github.com/astral-sh/uv) recommended (installs dependencies in seconds)

### 2. Clone and install

```bash
git clone https://github.com/jayasoruban/documind-rag-chatbot.git
cd documind-rag-chatbot

uv venv
uv pip install -e .
```

Or with classic pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and paste your OpenAI key:

```env
OPENAI_API_KEY=sk-...
```

### 4. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a document in the sidebar, click **Index documents**, then ask a question in the chat.

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← chat, upload, knowledge-base management
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────────┐
│  Document       │──→  │  Text Splitter       │  ← RecursiveCharacterTextSplitter
│  Loader         │      │  (1000 / 200 overlap)│
└─────────────────┘      └──────────┬──────────┘
                                    │
                                    ▼
┌─────────────────┐      ┌─────────────────────┐
│  OpenAI         │ ←→  │  ChromaDB            │  ← persistent local vector store
│  Embeddings     │      │  (text-embedding-3-small)
└─────────────────┘      └──────────┬──────────┘
                                    │
                            query + top-K chunks
                                    │
                                    ▼
┌─────────────────┐      ┌─────────────────────┐
│  Prompt         │──→  │  ChatOpenAI          │  ← streaming LCEL chain
│  Template       │      │  (gpt-4o-mini)      │
└─────────────────┘      └──────────┬──────────┘
                                    │
                                    ▼
                          cited answer + sources
```

### Module layout

```
documind-rag-chatbot/
├── app.py                      # Streamlit entry point
├── src/
│   ├── config.py               # Environment-driven settings
│   ├── document_loader.py      # Multi-format ingestion + chunking
│   ├── vector_store.py         # ChromaDB persistence
│   └── rag_chain.py            # LCEL retrieval + streaming
├── .streamlit/config.toml      # UI theming
├── Dockerfile                  # Container build
├── pyproject.toml              # Dependencies (uv / pip compatible)
└── .env.example                # Configuration template
```

---

## 🐳 Deploy

### Docker

```bash
docker build -t documind .
docker run -p 8501:8501 --env-file .env documind
```

### Render / Railway / Fly

Any container platform works. Set environment variables from `.env.example`, expose port `8501`, and point the start command at `streamlit run app.py --server.address=0.0.0.0 --server.port=8501`.

### Streamlit Community Cloud

1. Push to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → connect the repo.
3. Add `OPENAI_API_KEY` to **Secrets**.
4. Deploy.

---

## ⚙️ Configuration

All settings live in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key. |
| `CHAT_MODEL` | `gpt-4o-mini` | Default chat model. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model. |
| `CHUNK_SIZE` | `1000` | Characters per chunk. |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks. |
| `TOP_K` | `4` | Chunks retrieved per query. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store location on disk. |
| `COLLECTION_NAME` | `documind` | Chroma collection name. |

---

## 🧩 Extending DocuMind

This project is intentionally small so you can fork and extend it. Common next steps:

- **Swap the vector store** → Pinecone, Weaviate, Qdrant (change `vector_store.py`)
- **Add hybrid search** → BM25 + vector (LangChain has `EnsembleRetriever`)
- **Add a reranker** → Cohere rerank or a cross-encoder before the LLM
- **Support more formats** → HTML, PPTX, CSV (add loaders in `document_loader.py`)
- **Add authentication** → Streamlit Authenticator or NextAuth on a FastAPI variant
- **Add evaluation** → RAGAS or TruLens for answer quality metrics

---

## 📄 License

MIT © [Jayasoruban R](https://github.com/jayasoruban)

---

<div align="center">

**Need a custom RAG system for your business?** I build these for teams.
[Start a conversation →](https://www.linkedin.com/in/jayasoruban-r-67b35b1bb/)

</div>
