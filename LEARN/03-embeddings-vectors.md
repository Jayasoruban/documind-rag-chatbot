# Lesson 3 — Embeddings & Vector Stores

> ⏱ ~35 minutes · 📁 reference file: `src/vector_store.py`

If Lesson 2 was the most important *operational* concept, this is the most important *theoretical* one. Understanding embeddings and vector stores is the literal difference between someone who uses LangChain and someone who can actually debug a RAG system.

## 1. What is an embedding?

An embedding is a **list of numbers that represents the meaning of a piece of text**.

```
"The cat sat on the mat."
        ↓ embed with OpenAI text-embedding-3-small
[0.0123, -0.0456, 0.0789, ...]  ← 1536 numbers
```

That list of 1536 numbers is a **point in 1536-dimensional space**. (Yes, imagine 1536-dim space. Nobody can. You just trust the math.)

**The magic property:** texts with similar *meaning* produce similar *vectors* — even if they share zero words.

```
"The cat sat on the mat."       → vector A
"A feline rested on the rug."   → vector B ≈ A
"The stock market crashed."     → vector C (far from A and B)
```

How "similar" is measured: **cosine similarity** (the angle between vectors). Score ranges from −1 (opposite) to 1 (identical meaning). In practice, most real-world pairs score 0.6–0.9.

### Why this works (the one-paragraph intuition)

Embedding models are trained on billions of sentences with a signal that says "these two pieces of text appeared in similar contexts." Over training, the model arranges text in vector space so that *similar context* → *nearby vectors*. After training, new text gets placed in that same space by meaning. You don't need to understand the transformer internals — just trust that meaning → geometry.

## 2. OpenAI embeddings — the pragmatic default

From `src/vector_store.py`:

```14:18:src/vector_store.py
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
```

We use `text-embedding-3-small`. Specs:

| Property | Value |
|---|---|
| Dimensions | 1536 |
| Max input tokens | 8191 |
| Cost | $0.02 per 1M tokens |
| Quality | Near state-of-the-art for English |

**At this price**, embedding a 500-page PDF ≈ $0.005. You can re-embed a client's entire corpus for pennies.

### Alternatives you should know

| Embedder | When to pick it |
|---|---|
| `text-embedding-3-large` (OpenAI) | Higher recall on hard domains (legal, medical). 3072 dims. 6x more expensive. |
| `voyage-3` (Voyage AI) | Current state-of-the-art on benchmarks. Good for hard retrieval problems. |
| `embed-english-v3` (Cohere) | Solid, cheap. 1024 dims. |
| `all-MiniLM-L6-v2` (sentence-transformers) | **Runs locally, free, 384 dims.** Use when data cannot leave the client's environment. |
| `bge-large-en-v1.5` (BAAI) | Best open-source. Runs on a laptop. |
| `nomic-embed-text` | Local, 768 dims, good quality, fits in Ollama. |

**The embedder choice is a security/cost/quality decision**. You'll make it with every client.

## 3. Critical rule: same embedder on ingest and query

You **must** use the same embedding model for storing chunks and embedding questions. Different models put text in different vector spaces — distances between them are meaningless.

Our code enforces this: `OpenAIEmbeddings` is used in both `add_documents` (ingest) and by the retriever (query) via the same `get_embeddings()` factory. If a client decides to switch from OpenAI to Voyage, **you must re-embed everything**. Store the embedder name in metadata so you can detect drift.

## 4. What is a vector store?

A vector store is a **database optimized for two operations**:

1. `add(vectors, metadata)` — store a vector with some metadata attached.
2. `similarity_search(query_vector, k=4)` — find the `k` vectors closest to `query_vector`.

That's it. Everything else (filtering, deletion, indexing) is bonus.

### Why not just use PostgreSQL?

PostgreSQL can store vectors (with `pgvector` extension), but a naive `SELECT` with cosine similarity is O(N) — it compares the query to every vector in the table. With 10M chunks, that's slow.

Vector stores solve this with **Approximate Nearest Neighbor (ANN) indexes** — typically **HNSW** (Hierarchical Navigable Small World) or **IVF**. These trade a tiny bit of accuracy (~1%) for 100–1000x speedup. ChromaDB uses HNSW under the hood.

## 5. ChromaDB — what we're using

From `src/vector_store.py`:

```21:28:src/vector_store.py
def get_vector_store() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
    )
```

Key things to notice:

- **`collection_name`** — a namespace. Multiple collections can live in one Chroma instance (e.g., one per client in multi-tenant apps).
- **`embedding_function`** — Chroma knows how to embed. You pass `Document`s (raw text) to `add_documents`, and Chroma calls the embedder for you. Nice abstraction.
- **`persist_directory`** — Chroma writes to disk here. On next startup it reloads automatically. No database server to manage.

### How ChromaDB stores data

In the `./chroma_db/` folder after you ingest:

```
chroma_db/
├── chroma.sqlite3          ← metadata and IDs
└── <collection-uuid>/
    ├── data_level0.bin     ← HNSW graph (the fast index)
    ├── header.bin
    ├── length.bin
    └── link_lists.bin
```

The `.bin` files are the HNSW index — a graph structure where each node points to its "near neighbors." Queries walk the graph, always moving toward nodes closer to the query vector. Finds top-K in milliseconds even with millions of vectors.

### Why ChromaDB specifically

| Feature | Chroma | Pinecone | Weaviate | Qdrant |
|---|---|---|---|---|
| Local / self-hosted | ✅ | ❌ (cloud only) | ✅ | ✅ |
| Free tier | ✅ unlimited | 100K vectors | ✅ | ✅ |
| Multi-tenant collections | ✅ | ✅ | ✅ | ✅ |
| Hybrid search (BM25 + vector) | Partial | ✅ | ✅ | ✅ |
| Metadata filtering | ✅ | ✅ | ✅ | ✅ |
| Scale | ~1M vectors comfortably | Billions | Millions+ | Millions+ |

**Chroma is the best choice for MVPs, demos, and on-prem small/medium deployments.** When a client crosses ~1M vectors or needs high-concurrency multi-region, you migrate to Pinecone or Qdrant. The swap is ~20 lines of code — LangChain abstracts the interface.

## 6. The CRUD operations in our code

### Add
```31:37:src/vector_store.py
def add_documents(documents: list[Document]) -> int:
    """Embed and store documents. Returns the number of chunks added."""
    if not documents:
        return 0
    store = get_vector_store()
    store.add_documents(documents)
    return len(documents)
```
Under the hood: for each `Document`, Chroma embeds `page_content`, generates a UUID, and stores `{id, vector, metadata, raw_text}`. Persisted to disk immediately.

### List unique sources
```40:49:src/vector_store.py
def list_sources() -> list[str]:
    """List unique source filenames currently in the knowledge base."""
    store = get_vector_store()
    data = store.get(include=["metadatas"])
    seen: set[str] = set()
    for meta in data.get("metadatas", []) or []:
        src = (meta or {}).get("source") or (meta or {}).get("filename")
        if src:
            seen.add(str(src))
    return sorted(seen)
```
`store.get()` without arguments returns all records. We pull all metadatas and dedupe by source.

### Delete by source
```52:59:src/vector_store.py
def delete_by_source(source: str) -> int:
    """Remove all chunks belonging to one uploaded file."""
    store = get_vector_store()
    data = store.get(where={"source": source})
    ids = data.get("ids", []) or []
    if ids:
        store.delete(ids=ids)
    return len(ids)
```
This is **metadata filtering** in action. `where={"source": "refund-policy.pdf"}` tells Chroma to return only chunks matching that filter. We collect the IDs and delete them.

**This is the pattern for multi-tenant apps:** filter by `tenant_id` on every query so tenants never see each other's data.

## 7. How similarity search actually works (the mental model)

At query time:

```
query = "What's our refund policy?"
           ↓ embed
query_vec = [0.013, -0.045, 0.021, ...]

Chroma's HNSW index:
    1. Start at a random entry node in the graph.
    2. Look at the node's neighbors. Pick the one closest to query_vec.
    3. Repeat until no neighbor is closer.
    4. Collect top-K closest nodes visited.

Return:
    [(chunk_id, cosine_similarity, text, metadata), ...]
```

The entire search is typically **10–50 milliseconds** for a corpus of 100K chunks. Compare with keyword search (needs a separate BM25 index) or SQL LIKE scan (100–1000x slower).

### Cosine similarity vs Euclidean distance vs dot product

Three possible metrics. **OpenAI embeddings are normalized to length 1**, so cosine similarity = dot product. Chroma's default cosine is fine — don't change it.

Rule: if you ever build your own embedder (e.g., from HuggingFace), call `.normalize()` on your vectors first. It makes the choice of metric irrelevant and improves consistency.

## 8. Remember this

- **Embedding = meaning-as-numbers.** Similar meanings → similar vectors. 1536 dims for `text-embedding-3-small`.
- **Use the same embedder for ingest and query.** Different models = different spaces = broken retrieval.
- **Vector stores are databases with two operations**: add & similarity_search. Fast because of ANN indexes (HNSW).
- **ChromaDB is the right default for MVPs.** Swap to Pinecone/Qdrant at scale.
- **Metadata filtering is how you do multi-tenancy, deletes, and selective search.**

## 9. Interview Q&A

**Q1: "How do embeddings work?"**
> Embedding models are neural networks (transformers) trained to put similar pieces of text close together in vector space. Training signal: pairs of sentences that appeared in related contexts should produce similar vectors. At inference time, the model converts any new text into a fixed-length vector — 1536 floats for OpenAI's small model. Similarity between two texts is the cosine of the angle between their vectors.

**Q2: "What happens if I use a different embedder at query time than I used at ingest?"**
> Your retrieval returns garbage. Different models produce different vector spaces — distances are meaningless across them. The fix is to re-embed the entire corpus with the new model. In production you version this: store `embedder_name` and `embedder_version` in metadata, and run a migration when changing.

**Q3: "When would you switch from Chroma to Pinecone?"**
> Three reasons: (1) corpus larger than a few million vectors, (2) need for high concurrency / multi-region replication, (3) need for managed service with uptime SLAs. For anything under 1M chunks on a single tenant, Chroma is faster to set up and free.

**Q4: "How do you do hybrid search?"**
> Run a BM25 keyword search and a vector search in parallel, then fuse the results using **Reciprocal Rank Fusion (RRF)** — each doc gets a score based on its rank in each list. Weaviate and Qdrant have this built in; in Chroma you'd bolt on `rank_bm25`. Hybrid catches both exact terms (SKUs, names) and semantic paraphrases, and it's the standard for production search.

**Q5: "How do you improve retrieval quality when it's bad?"**
> Diagnosis flow: (1) log the retrieved chunks and inspect them — is the right chunk *present* but buried? → switch to a reranker (Cohere Rerank, Voyage Rerank). Is the right chunk *not in the top-50* at all? → chunking or embedding problem. Tweak chunk size or try a better embedder. (2) Add query rewriting — rewrite the user question into 2–3 search-optimized variants. (3) Add metadata filters. In my experience these 3 fix 80% of retrieval issues.

## 10. Exercise

1. **Inspect a real embedding.** Open a Python REPL inside the venv:
   ```python
   from src.vector_store import get_embeddings
   emb = get_embeddings()
   vec = emb.embed_query("What's our refund policy?")
   print(len(vec), vec[:5])
   ```
   Confirm you get a 1536-length list. Look at the first 5 numbers.

2. **Measure similarity by hand.**
   ```python
   import numpy as np
   a = np.array(emb.embed_query("refund policy for enterprise customers"))
   b = np.array(emb.embed_query("return rules for big company clients"))
   c = np.array(emb.embed_query("what's the weather today"))
   print("similar:",  a @ b)   # should be ~0.6–0.8
   print("unrelated:", a @ c)  # should be ~0.1–0.3
   ```
   This is cosine similarity (because OpenAI vectors are normalized, dot product = cosine).

3. **Explore the Chroma files.** After ingesting one document, `ls chroma_db/`. Look at the sqlite file with `sqlite3 chroma_db/chroma.sqlite3 ".tables"`. Understand what's stored where.

Once you've touched real embeddings with your own hands, the abstraction disappears and it becomes obvious.

---

[← Lesson 2](./02-ingestion-chunking.md) &nbsp;·&nbsp; [Next: Lesson 4 — Retrieval & LCEL Chains →](./04-retrieval-lcel.md)
