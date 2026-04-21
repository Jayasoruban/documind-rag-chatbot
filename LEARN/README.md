# 📘 Learning Notes

These are my personal learning notes while building DocuMind — structured as lessons, one concept at a time. Each lesson is grounded in the actual code in this repo.

> If you're a recruiter or client reading this: these notes are here because I believe shipping code I don't deeply understand is a liability. Each file here walks through a specific concept using the actual implementation in this project.

## Lessons

1. [**The Big Picture**](./01-big-picture.md) — What RAG is, why it exists, how DocuMind fits together.
2. [**Ingestion & Chunking**](./02-ingestion-chunking.md) — Loaders, chunk size trade-offs, metadata, citations.
3. [**Embeddings & Vector Stores**](./03-embeddings-vectors.md) — Semantic search, ChromaDB, cosine similarity.
4. [**Retrieval & LCEL Chains**](./04-retrieval-lcel.md) — Top-K retrieval, prompts, the pipe pattern.
5. [**Streaming & Streamlit UI**](./05-streaming-ui.md) — Token streaming, session state, chat patterns.
6. [**Production & What's Next**](./06-production-next.md) — Deployment, eval, and the path to LangGraph/CrewAI/MCP.

## How to use these notes

- Read a lesson top to bottom (15–40 min each).
- Do the exercise at the end before moving on — that's what makes it stick.
- Keep the relevant source file open in a second window while reading.
- Mark questions in the margins; the **Interview Q&A** section at the end of each lesson is exactly the kind of thing clients and interviewers will ask.
