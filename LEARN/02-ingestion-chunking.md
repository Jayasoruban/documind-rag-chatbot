# Lesson 2 — Ingestion & Chunking

> ⏱ ~30 minutes · 📁 reference file: `src/document_loader.py`

The quality of a RAG system is capped by the quality of its ingestion pipeline. Retrieval can't find what ingestion didn't prepare well. This is the single biggest place where production RAG succeeds or fails.

## 1. What does "ingestion" mean?

Ingestion is the one-time process of turning a raw file into something searchable:

```
raw file  →  text  →  chunks  →  embeddings  →  vector store
         (loader)  (splitter)   (covered in Lesson 3)
```

This lesson covers the first two arrows — turning files into well-sized chunks of text with useful metadata.

## 2. Loaders: why we need one per file type

PDFs, Word docs, Markdown files, and plain text each have their own parsing rules. PDFs in particular are a *nightmare* — text layout, columns, tables, images, scanned pages. LangChain wraps this mess in consistent loaders that all return `Document` objects.

From `src/document_loader.py`:

```20:31:src/document_loader.py
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
```

**Key point:** every loader returns `list[Document]`. A `Document` is just:

```python
Document(
    page_content="the actual text",
    metadata={"source": "refund-policy.pdf", "page": 3},
)
```

That's it. Text + metadata. Everything downstream works on this simple shape.

### Which loader handles what

| Loader | Best for | Gotchas |
|---|---|---|
| `PyPDFLoader` | Most PDFs. Splits by page automatically. | Fails on scanned PDFs — needs OCR. |
| `TextLoader` | `.txt`, `.log`, raw files. | Needs explicit `encoding`. UTF-8 is safe default. |
| `UnstructuredMarkdownLoader` | `.md` files. Preserves structure. | Requires `unstructured` package (heavy dep). |
| `Docx2txtLoader` | `.docx` (modern Word). | Strips formatting. Tables come out flattened. |

**Production alternatives you'll encounter in client work:**

- **`Unstructured.io`** — the industrial-strength option. Handles scanned PDFs, complex layouts, tables. Has an API. Use for enterprise clients.
- **`LlamaParse`** (from LlamaIndex) — best-in-class for complex PDFs with tables. Paid.
- **`pdfplumber`** — better text extraction than PyPDF for multi-column layouts.
- **Azure Document Intelligence / AWS Textract** — for scanned documents and forms.

## 3. Why we chunk (and why it matters more than you think)

Imagine you upload a 300-page contract. You have three bad options if you *don't* chunk:

1. **Embed the whole document as one vector.** Then all 300 pages map to a single point in vector space. A question about page 273 gets the same answer as a question about page 5. Terrible recall.
2. **Feed the whole document to the LLM as context.** 300 pages ≈ 150K tokens. Overflows most models. Even when it fits, the LLM pays attention poorly to very long context ("lost in the middle" problem).
3. **Search by page.** Naive chunks lose mid-page context and split sentences weirdly.

**Chunking = break documents into pieces that are (a) small enough to embed precisely, (b) big enough to be meaningful on their own.**

## 4. Chunk size — the single most important knob

From `src/document_loader.py`:

```46:59:src/document_loader.py
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
```

### The trade-off

```
        chunk too small                 chunk too big
        (e.g., 100 chars)                (e.g., 4000 chars)
             │                                │
             ▼                                ▼
   "fragments lack context"        "retrieval gets noisy"
   "LLM gets lots of chunks"       "embeddings average out meaning"
   "cost goes up"                  "LLM reads junk alongside gold"
```

**Rules of thumb from production:**

- **FAQs / Q&A docs:** ~500 chars. Each Q&A pair is self-contained.
- **Prose / articles / manuals:** ~1000 chars. DocuMind's default.
- **Code / technical docs:** ~1500 chars with structural separators.
- **Research papers / contracts:** ~2000 chars. Dense, concept-heavy.

We picked **1000 characters** because DocuMind is a general-purpose tool. A client with specific document types would tune this.

### Overlap: the duct tape of chunking

```
Chunk 1: "...the customer may request a refund within 30"
Chunk 2: "within 30 days of purchase, provided..."
         ^^^^^^^^^^^^^ overlap of 200 chars ^^^^^^^^^^^^^
```

Without overlap, sentences that straddle chunk boundaries lose meaning. We use **200 chars overlap (20% of chunk size)** — the standard ratio.

**Cost of overlap:** you embed slightly more text (20% more chunks ≈ 20% more cost). Worth it.

## 5. Why `RecursiveCharacterTextSplitter` and not naive splitting

Naive splitter: `text[i*1000:(i+1)*1000]`. Awful — cuts mid-word, mid-sentence, mid-paragraph.

Our splitter:

```python
separators=["\n\n", "\n", ". ", " ", ""]
```

Tries to split in that priority order. The algorithm:

1. Try to split on `\n\n` (paragraph breaks). If all resulting pieces are ≤ `chunk_size`, done.
2. If any piece is still too big, recursively split *that* piece on `\n` (line breaks).
3. Still too big? Try `. ` (sentence ends).
4. Still too big? Fall back to spaces, then characters.

**Result:** chunks break at the most semantically natural boundary available. This is one of the simplest yet most impactful ideas in RAG.

For code, swap in `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` — it uses Python-aware separators like `\nclass `, `\ndef `, `\n\n`.

## 6. Metadata: the secret to good UX

Remember this from the loader:

```34:43:src/document_loader.py
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
```

We stamp every `Document` with `source` and `filename`. The splitter **copies metadata to every chunk** automatically. That's why `PyPDFLoader`'s automatic `page` metadata survives the split and ends up on every chunk from that page.

**What metadata unlocks:**

- **Citations.** When we show "Source: refund-policy.pdf (page 3)", that comes from metadata attached to the retrieved chunk.
- **Filtering.** `store.get(where={"source": "refund-policy.pdf"})` — used in our `delete_by_source()`.
- **Security.** In a multi-tenant SaaS, stamp every chunk with `tenant_id` and filter retrievals. Bare minimum data isolation.
- **Freshness.** Stamp `ingested_at` and boost recent chunks at query time.

**Production tip:** add as much metadata as you can afford at ingestion time. It's nearly free, and you'll always wish you had more later.

## 7. Remember this

- **Loader** = turns a file into `list[Document]`. One `Document` = text + metadata.
- **Splitter** = turns big `Document`s into small overlapping chunks.
- **Chunk size & overlap are trade-offs.** 1000 / 200 is a safe default.
- **`RecursiveCharacterTextSplitter`** splits on natural boundaries (paragraphs → lines → sentences).
- **Metadata travels with every chunk** automatically. Use this for citations, filtering, security.

## 8. Interview Q&A

**Q1: "How do you decide chunk size for a new document type?"**
> Start with 1000/200. Then measure — upload 5 documents, run 20 realistic queries, and manually grade retrieval quality. If the LLM is missing context, increase chunk size. If retrieval returns off-topic passages, decrease it. For short-form content like FAQs, go smaller; for dense long-form content, go bigger.

**Q2: "What's chunk overlap and why 20%?"**
> Overlap is the number of characters shared between consecutive chunks. It prevents semantic units (sentences, list items) from being split across chunks. 20% is industry standard — enough to catch cross-boundary info without exploding storage costs. Below 10% you start missing context; above 30% it's wasteful.

**Q3: "What happens with scanned PDFs?"**
> `PyPDFLoader` returns empty strings because there's no text layer. You need OCR. For production I'd use `Unstructured.io` with OCR enabled, or Azure Document Intelligence for forms. For a quick solution, `pytesseract` + `pdf2image` locally.

**Q4: "Why do you add a `source` field manually when PyPDF already sets one?"**
> `PyPDFLoader` sets `source` to the full filesystem path, which is messy for UI display and leaks server paths in citations. Overriding with just the filename gives clean citations and decouples metadata from where the file physically lives on the server.

**Q5: "How do you handle updates to documents?"**
> The cleanest approach: delete all chunks with that `source`, then re-ingest. Our `delete_by_source()` in `vector_store.py` does exactly this. Smarter approaches use content hashes — only re-embed chunks whose text actually changed. We don't do that here because it's overkill for an MVP.

## 9. Exercise

1. **Try different chunk sizes.** In `.env`, set `CHUNK_SIZE=300` then `CHUNK_SIZE=2000`. Upload the same PDF both times. Ask the same 3 questions. Which size gives better answers?
2. **Add a `chunk_index` metadata field.** In `chunk_documents()`, after splitting, loop over the chunks and add `chunk["chunk_index"] = i`. Then modify `rag_chain.py`'s citation formatter to show it. This is how production systems give precise deep-links.
3. **Bonus:** replace `PyPDFLoader` with `pdfplumber` for better multi-column handling. Ingest a two-column academic paper and compare.

Once you've done at least exercise 1, you've internalized chunking. Move to Lesson 3.

---

[← Lesson 1](./01-big-picture.md) &nbsp;·&nbsp; [Next: Lesson 3 — Embeddings & Vector Stores →](./03-embeddings-vectors.md)
