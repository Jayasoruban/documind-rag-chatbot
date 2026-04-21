"""
Document ingestion — step 1 of the RAG pipeline.

This file is responsible for:
  (a) turning a raw file (PDF / TXT / MD / DOCX) into LangChain Document objects, and
  (b) splitting those Documents into small overlapping chunks sized for embedding.

See LEARN/02-ingestion-chunking.md for the deep dive on *why* we chunk
and how chunk size / overlap affect retrieval quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# LangChain ships a different loader for each file type. Each one handles
# the messy details of parsing that format (PDF text extraction, DOCX zip
# structure, markdown AST) and returns a consistent list[Document] shape.
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)

# A LangChain Document is a tiny dataclass: {page_content: str, metadata: dict}.
# Everything downstream (the splitter, the vector store, the LLM prompt builder)
# operates on this uniform shape — that's the whole point of the abstraction.
from langchain_core.documents import Document

# RecursiveCharacterTextSplitter splits text at NATURAL boundaries (paragraphs,
# then lines, then sentences, then words) instead of just slicing at fixed
# character counts. This preserves semantic units and is a big quality win
# compared to naive splitting.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The UI uses this to filter the file-upload widget to only these extensions.
# Keeping it at module level means app.py imports one canonical list — no drift.
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}


def load_file(path: Path) -> list[Document]:
    """Turn one file into LangChain Documents by picking the right loader.

    Each loader knows how to parse its format. PyPDFLoader, for example,
    splits a PDF into one Document per page and auto-populates
    metadata={"source": path, "page": N}. That per-page metadata becomes
    critical later when we render citations ("refund-policy.pdf page 3").

    Raises ValueError for unsupported extensions so the UI can show a clean
    error message instead of silently dropping files.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        # PyPDFLoader is fast and works for ~90% of digital PDFs. For scanned
        # PDFs (image-based, no text layer), it returns empty strings — in
        # production you'd swap in Unstructured or Azure Document Intelligence
        # with OCR enabled.
        return PyPDFLoader(str(path)).load()
    if ext in {".txt"}:
        # Explicit encoding avoids cryptic decode errors when users upload
        # files created on Windows with non-UTF-8 encoding.
        return TextLoader(str(path), encoding="utf-8").load()
    if ext in {".md", ".markdown"}:
        # UnstructuredMarkdownLoader preserves the document structure
        # (headings, lists) better than reading markdown as plain text.
        return UnstructuredMarkdownLoader(str(path)).load()
    if ext == ".docx":
        # Docx2txtLoader strips formatting but handles modern Word files
        # (zip + XML under the hood). Good enough for most business docs.
        return Docx2txtLoader(str(path)).load()
    raise ValueError(f"Unsupported file type: {ext}")


def load_files(paths: Iterable[Path]) -> list[Document]:
    """Load many files and stamp a clean filename onto each Document's metadata.

    Why the metadata stamping matters:
      - Loaders set `source` to the full filesystem path (ugly, leaks server
        paths in citations). We override with just the filename.
      - We ALSO set `filename` as a stable field. Some loaders don't set
        `source` consistently; having both keys gives the RAG chain a
        reliable fallback when building citation labels.
      - Any metadata added here automatically propagates to every chunk
        produced by the splitter below — we never need to re-stamp later.
    """
    all_docs: list[Document] = []
    for path in paths:
        docs = load_file(path)
        for doc in docs:
            # setdefault leaves the loader's source alone if it already set one,
            # so we don't clobber e.g. PyPDFLoader's useful page-level source.
            doc.metadata.setdefault("source", path.name)
            # filename is always the clean display name — used for UI citations.
            doc.metadata["filename"] = path.name
        all_docs.extend(docs)
    return all_docs


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split Documents into small overlapping chunks ready for embedding.

    Why we chunk at all:
      - Embeddings work best on focused text. A 100-page document as one
        embedding produces a generic "this document is about law" vector
        that matches everything and nothing.
      - LLM context windows are finite. We can only feed the LLM a handful
        of chunks per query.
      - Semantic granularity — users ask about specific passages, not
        whole documents.

    Why 1000 chars / 200 overlap (the defaults):
      - Small enough that each chunk is topically coherent (~1 paragraph)
      - Large enough to carry useful context (not fragmentary)
      - 20% overlap catches sentences that straddle chunk boundaries
      A client with specialized docs (FAQs → smaller, research papers →
      bigger) would tune these values per project.

    The `separators` list is the key trick. The splitter tries them in
    priority order: paragraphs first, then line breaks, then sentence ends,
    then spaces, then finally individual characters. This way chunks break
    at the most semantically natural boundary available.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Measure in characters (default). Could swap for token count for tighter cost control.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # split_documents() preserves each input Document's metadata on every
    # resulting chunk — this is how `source`, `filename`, and PyPDF's
    # auto-populated `page` all survive splitting and reach the UI.
    return splitter.split_documents(documents)
