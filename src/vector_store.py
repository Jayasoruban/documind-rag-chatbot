"""Persistent Chroma vector store with simple collection management."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vector_store() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
    )


def add_documents(documents: list[Document]) -> int:
    """Embed and store documents. Returns the number of chunks added."""
    if not documents:
        return 0
    store = get_vector_store()
    store.add_documents(documents)
    return len(documents)


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


def delete_by_source(source: str) -> int:
    """Remove all chunks belonging to one uploaded file."""
    store = get_vector_store()
    data = store.get(where={"source": source})
    ids = data.get("ids", []) or []
    if ids:
        store.delete(ids=ids)
    return len(ids)


def reset_store() -> None:
    """Drop every chunk from the collection (preserves the directory)."""
    store = get_vector_store()
    all_ids = store.get().get("ids", []) or []
    if all_ids:
        store.delete(ids=all_ids)


def count_chunks() -> int:
    store = get_vector_store()
    return len(store.get().get("ids", []) or [])
