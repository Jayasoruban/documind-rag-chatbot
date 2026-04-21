"""Document ingestion: load PDFs, text, markdown, and Word documents into LangChain Documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}


def load_file(path: Path) -> list[Document]:
    """Dispatch to the correct LangChain loader based on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext in {".txt"}:
        return TextLoader(str(path), encoding="utf-8").load()
    if ext in {".md", ".markdown"}:
        return UnstructuredMarkdownLoader(str(path)).load()
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    raise ValueError(f"Unsupported file type: {ext}")


def load_files(paths: Iterable[Path]) -> list[Document]:
    """Load many files, preserving the original filename in the metadata."""
    all_docs: list[Document] = []
    for path in paths:
        docs = load_file(path)
        for doc in docs:
            doc.metadata.setdefault("source", path.name)
            doc.metadata["filename"] = path.name
        all_docs.extend(docs)
    return all_docs


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks sized for embedding models."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
