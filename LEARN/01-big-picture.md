# Lesson 1 — The Big Picture

> ⏱ ~25 minutes · 📁 reference files: all of them

## 1. What problem does RAG solve?

LLMs like GPT-4 and Claude are impressive, but they have **three hard limits**:

1. **Knowledge cutoff.** They only know what was in their training data. GPT-4o-mini doesn't know about your company's Q4 2026 sales report.
2. **No private data.** They never saw your internal docs, contracts, codebases, or emails.
3. **Hallucination.** When they don't know something, they *invent* plausible-sounding answers confidently. That's worse than saying "I don't know."

You could solve this by **retraining or fine-tuning the model** on your data — but that costs $10K–$100K, takes weeks, and has to be redone every time your data changes.

**RAG (Retrieval-Augmented Generation)** is the cheaper, faster, more accurate alternative.

## 2. What RAG actually does

Think of it like this:

> **RAG is Ctrl-F for LLMs.**
> When a user asks a question, we first **search** the user's documents for the most relevant chunks, then **hand those chunks to the LLM along with the question**, and ask the LLM to answer using only that context.

Flow:

```
User asks: "What's our refund policy for enterprise customers?"

Step 1 — Retrieve:
  Search the document store for chunks most similar to the question.
  → Found 4 chunks from "enterprise-contract.pdf" pages 3–7.

Step 2 — Augment:
  Build a prompt that contains:
    (a) instructions to the LLM
    (b) the retrieved chunks as "context"
    (c) the user's question

Step 3 — Generate:
  Send the prompt to GPT-4o-mini.
  Get a natural-language answer grounded in the retrieved chunks.
  Show the sources so the user can verify.
```

**Key insight:** the LLM is not "learning" your data. It's reading your data freshly each time. You can update, add, or remove docs anytime — no retraining needed.

## 3. DocuMind's architecture

Here's our project mapped to the 3 RAG stages:

```
┌────────────────────── INGESTION (one-time per doc) ─────────────────────┐
│                                                                          │
│  User uploads PDF/DOCX/TXT/MD                                            │
│           ↓                                                              │
│  [document_loader.py]  Parse into LangChain Documents                    │
│           ↓                                                              │
│  [document_loader.py]  Split into overlapping chunks (~1000 chars)       │
│           ↓                                                              │
│  [vector_store.py]     Embed each chunk with OpenAI embeddings           │
│           ↓                                                              │
│  [vector_store.py]     Store vectors in ChromaDB (on disk)               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌────────────────────── QUERY (every question) ───────────────────────────┐
│                                                                          │
│  User types question                                                     │
│           ↓                                                              │
│  [rag_chain.py]   Embed the question                                     │
│           ↓                                                              │
│  [rag_chain.py]   Search ChromaDB for top-K most similar chunks          │
│           ↓                                                              │
│  [rag_chain.py]   Format chunks + question into a prompt                 │
│           ↓                                                              │
│  [rag_chain.py]   Stream answer from GPT-4o-mini                         │
│           ↓                                                              │
│  [app.py]         Render answer + expandable source citations            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

Every file in `src/` maps to one stage. This isn't accidental — it's how most production RAG systems are structured. When you build Project 2 (multi-agent research), you'll reuse this same mental model.

## 4. Key vocabulary (memorize these)

You'll hear these in every AI freelance interview:

| Term | What it means |
|---|---|
| **Chunk** | A small piece of a document (200–2000 chars). We chunk because LLM context windows are finite and embeddings work better on focused text. |
| **Embedding** | A list of numbers (e.g., 1536 floats) that represents the *meaning* of a piece of text. Similar meanings → similar numbers. |
| **Vector store** | A database optimized for storing embeddings and finding similar ones fast. (ChromaDB, Pinecone, Weaviate, Qdrant.) |
| **Semantic search** | Searching by meaning instead of exact keywords. "car" and "automobile" score as similar. |
| **Retriever** | The component that, given a question, fetches the top-K most relevant chunks. |
| **Top-K** | How many chunks we retrieve. We use 4. Too few = missed context. Too many = distraction + cost. |
| **Context window** | How many tokens an LLM can see at once. GPT-4o-mini = 128K tokens. |
| **Hallucination** | When the LLM makes up facts. RAG reduces this by forcing the LLM to answer from provided context. |
| **Grounding** | Tying the LLM's output to retrieved evidence. "Grounded answers" = answers with citations. |
| **LCEL** | LangChain Expression Language — the `prompt | llm | parser` pipe syntax. |

## 5. Why this architecture (design decisions clients will ask about)

**Why ChromaDB and not Pinecone?**
- Free, runs locally, persists to disk. Good for MVPs and on-prem clients.
- Pinecone is better for multi-million-vector production at scale ($70+/mo).
- DocuMind uses ChromaDB; for enterprise clients we'd swap in Pinecone (takes ~20 lines of code).

**Why OpenAI embeddings (`text-embedding-3-small`)?**
- 1536 dimensions, $0.02 per 1M tokens — cheapest tier with excellent quality.
- Alternatives: Cohere, Voyage, HuggingFace sentence-transformers (free, runs locally).
- For regulated industries (healthcare, legal), we'd swap in a local embedder so data never leaves the company.

**Why `gpt-4o-mini` as default?**
- 10x cheaper than `gpt-4o`, 80% of the quality for RAG tasks (where the LLM is summarizing retrieved context, not reasoning hard).
- We expose a model dropdown so users can switch to `gpt-4o` or `gpt-4.1` for tough questions.

**Why Streamlit and not Next.js?**
- Streamlit is 10x faster for demos and internal tools. No frontend code.
- For customer-facing products, you'd replace Streamlit with Next.js + FastAPI. The `src/` layer stays untouched — that's why we separated it.

## 6. Remember this

- **RAG = Retrieve → Augment → Generate.** Always those three steps.
- The LLM never *learns* your data; it reads it fresh every time.
- DocuMind has two flows: **ingestion** (runs once per upload) and **query** (runs every question).
- The whole point of the `src/` layer is to keep the Streamlit UI **replaceable**. If a client wants a Slack bot, you reuse `src/` and swap `app.py`.
- Every design choice (vector store, embedding model, LLM) is a **trade-off of cost, quality, privacy, and latency**. Knowing the trade-offs is what makes you senior.

## 7. Interview Q&A

**Q1: "Why would you use RAG instead of fine-tuning?"**
> Fine-tuning costs thousands of dollars and has to be redone when data changes. RAG lets the LLM read fresh data at query time — cheaper, faster to iterate, and you always have an audit trail (the retrieved chunks). Fine-tuning is better when you need the model to adopt a specific *style* or *domain vocabulary*, not just know specific facts.

**Q2: "What's the difference between semantic search and keyword search?"**
> Keyword search matches exact strings. Semantic search matches meaning. "car broke down" and "vehicle malfunction" don't share keywords but should return the same results. Semantic search does this by comparing embeddings — vectors that capture meaning. The best production systems use **hybrid search** (both), with keyword search catching exact terms and semantic search catching paraphrases.

**Q3: "How do you handle hallucinations in a RAG system?"**
> Three layers: (1) strict system prompts that tell the LLM to only answer from context and say "I don't know" otherwise, (2) always show sources so users can verify, (3) evaluation pipelines that measure answer faithfulness against retrieved chunks (tools like RAGAS). DocuMind's system prompt in `rag_chain.py` is the first layer.

**Q4: "When would you NOT use RAG?"**
> (a) When the task is reasoning or math, not fact lookup. (b) When the total corpus fits in the context window — just paste it in directly. (c) When latency matters more than accuracy (a raw LLM call is ~500ms faster than retrieve-then-generate). (d) When data is structured (use SQL or a knowledge graph instead).

**Q5: "What's the biggest hidden cost in a RAG system?"**
> Embeddings at scale. A 10,000-page corpus = ~5M tokens ≈ $100 to embed once. But re-embedding when you upgrade models or tweak chunking is where teams get stuck. Always version your embeddings and store the embedding model name in metadata.

## 8. Exercise

Before moving to Lesson 2:

1. Open `app.py` and find where `stream_answer` is called.
2. Trace the call into `src/rag_chain.py` → into `retrieve()` → into `src/vector_store.py` → into `get_vector_store()`.
3. **Write a 5-line comment at the top of `app.py`** describing the flow of a single user question through the codebase (for your own reference).

If you can do that without referring back to this lesson, you've got the Big Picture. On to Lesson 2.

---

[← Back to index](./README.md) &nbsp;·&nbsp; [Next: Lesson 2 — Ingestion & Chunking →](./02-ingestion-chunking.md)
