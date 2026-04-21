"""
The vector store layer — step 2 and 3 of the RAG pipeline.

Responsibilities:
  - Embedding: turn text into 1536-dimensional vectors with OpenAI.
  - Storage: persist those vectors in a local ChromaDB on disk.
  - Retrieval helpers: list, delete, count, reset.

Why ChromaDB specifically:
  - Runs locally. No server to manage. Perfect for MVPs and on-prem.
  - Persists to disk — survives restarts without re-embedding everything.
  - Good up to ~1M vectors. For bigger scale we'd swap in Pinecone/Qdrant
    (LangChain's Chroma class has the same interface, so the swap is tiny).

See LEARN/03-embeddings-vectors.md for the deep dive on embeddings,
cosine similarity, HNSW indexes, and alternatives.
"""

from __future__ import annotations

from pathlib import Path

# LangChain's thin wrapper around the chromadb python library. We could use
# chromadb directly, but the LangChain wrapper gives us interface parity with
# Pinecone, Weaviate, Qdrant, etc. — swap one import to migrate.
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import settings


def get_embeddings() -> OpenAIEmbeddings:
    """Build an OpenAI embeddings client configured from settings.

    CRITICAL RULE: the embedder used here MUST match the embedder used
    when documents were ingested. If you switch from text-embedding-3-small
    to text-embedding-3-large, every stored vector becomes meaningless
    (they live in a different 3072-dim space, not our 1536-dim space)
    and you'll need to re-embed the entire corpus.

    That's why both `add_documents` (ingest path) and the retriever below
    (query path) obtain their embedder through this same factory function.
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vector_store() -> Chroma:
    """Open (or lazily create) the persistent Chroma collection.

    Two important properties:
      1. Calling this when the directory doesn't exist is safe — Chroma
         creates the sqlite + HNSW files on first write.
      2. Calling it again on subsequent runs reloads the existing index
         instantly (no re-embedding needed). This is the whole point of
         `persist_directory`: DocuMind remembers what you uploaded last
         week.

    Why we build a fresh Chroma object on every call instead of caching:
      - ChromaDB's client is lightweight to construct.
      - Avoids stale state if you ever run this inside a long-lived server.
      - In practice most calls happen once per user action, so there's no
        real overhead.
    """
    # exist_ok=True → no error if the directory already exists; parents=True →
    # creates any missing parent directories along the way.
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        # The collection is ChromaDB's equivalent of a table or namespace.
        # For multi-tenant SaaS later you'd have one collection per client.
        collection_name=settings.collection_name,
        # Passing the embedder in means Chroma will call embed_documents()
        # for us whenever we add Documents. Nice abstraction — we never
        # touch raw vectors ourselves.
        embedding_function=get_embeddings(),
        # This is the disk location. Chroma writes both a sqlite file
        # (metadata + IDs) and binary HNSW index files (the fast search
        # structure) into this folder.
        persist_directory=str(settings.chroma_persist_dir),
    )


def add_documents(documents: list[Document]) -> int:
    """Embed and persist a batch of chunks. Returns how many were added.

    Under the hood Chroma will:
      1. Call OpenAIEmbeddings.embed_documents() with the chunks' text.
         (One API call per ~100 chunks — batched automatically.)
      2. Generate a UUID for each chunk.
      3. Write {id, vector, metadata, raw_text} into sqlite and update
         the HNSW index.
      4. Flush to disk so the data survives a restart.

    Costs pennies. Embedding 500 pages ≈ $0.005.
    """
    # Short-circuit on empty input so we don't make a pointless network call.
    if not documents:
        return 0
    store = get_vector_store()
    store.add_documents(documents)
    return len(documents)


def list_sources() -> list[str]:
    """Return the unique source filenames currently in the knowledge base.

    Used by the sidebar to show the user "here's what I know about."

    Approach: Chroma's `get()` with no filter returns EVERY record. We pull
    just the metadata (not the raw text — cheaper) and dedupe the 'source'
    or 'filename' field across all chunks. The result is one entry per
    uploaded file, not one per chunk.

    For a corpus of a million chunks this would be slow; we'd add a
    separate 'files' sqlite table in that world. For an MVP it's fine.
    """
    store = get_vector_store()
    # include=["metadatas"] keeps the response small by skipping the raw
    # chunk text we don't need here.
    data = store.get(include=["metadatas"])
    seen: set[str] = set()
    for meta in data.get("metadatas", []) or []:
        # Chunks from different loader types may have 'source' or 'filename'
        # or both. Try both keys and take the first available.
        src = (meta or {}).get("source") or (meta or {}).get("filename")
        if src:
            seen.add(str(src))
    return sorted(seen)


def delete_by_source(source: str) -> int:
    """Remove every chunk belonging to one uploaded file.

    This is metadata filtering in action — `where={"source": source}`
    tells Chroma to only return chunks whose metadata matches that key/value.

    The same pattern is how you'd do multi-tenant isolation in production:
    every chunk gets stamped with a `tenant_id`, and every query includes
    `where={"tenant_id": current_user.tenant_id}`. Tenants can never see
    each other's data because the filter applies before retrieval.
    """
    store = get_vector_store()
    data = store.get(where={"source": source})
    ids = data.get("ids", []) or []
    if ids:
        # We delete by ID, not by filter. Chroma supports both but deleting
        # by ID is more predictable (no chance of accidentally catching
        # chunks we didn't mean to with a too-broad filter).
        store.delete(ids=ids)
    return len(ids)


def reset_store() -> None:
    """Wipe every chunk from the collection (keep the folder itself).

    We don't delete the persist_directory because Chroma keeps connection
    state tied to it; nuking the folder mid-run causes hangs. Deleting by
    IDs is cleaner and returns the collection to an empty-but-valid state.
    """
    store = get_vector_store()
    all_ids = store.get().get("ids", []) or []
    if all_ids:
        store.delete(ids=all_ids)


def count_chunks() -> int:
    """Total number of chunks currently stored.

    Used by the UI to show "3 files · 127 chunks" stats, and to decide
    whether to block queries when the knowledge base is empty.
    """
    store = get_vector_store()
    # .get() with no args returns everything; we just count the ID list
    # rather than pulling vectors/texts we don't need.
    return len(store.get().get("ids", []) or [])
