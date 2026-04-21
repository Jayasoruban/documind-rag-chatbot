"""
The RAG chain — step 4 of the pipeline, where retrieval meets the LLM.

Given a user question this module:
  1. Retrieves the top-K most relevant chunks from the vector store.
  2. Formats them into a prompt with clear citation markers.
  3. Sends everything to the LLM (optionally streaming the response).
  4. Returns the answer plus the source chunks so the UI can show citations.

It exposes two entry points:
  - answer(question) → RagAnswer       (blocks until complete)
  - stream_answer(question) → generator (yields tokens as they arrive)

See LEARN/04-retrieval-lcel.md for the deep dive on LCEL (the `|` syntax),
prompts, and why this architecture stops hallucinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from langchain_core.documents import Document

# ChatPromptTemplate lets us build multi-role prompts (system + user + ai)
# with variable substitution — more powerful than raw f-strings because
# it knows about OpenAI's chat message format.
from langchain_core.prompts import ChatPromptTemplate

# LangChain's wrapper around OpenAI's chat API. Critical feature: it
# implements the LCEL Runnable interface, so it composes cleanly with
# prompts and parsers via the pipe operator.
from langchain_openai import ChatOpenAI

from .config import settings
from .vector_store import get_vector_store


# ------------------------------------------------------------------------
# PROMPTS
# ------------------------------------------------------------------------
# The system prompt is the single biggest lever for RAG quality. Every
# line below is deliberate:
#   - "strictly using the provided context" → anti-hallucination clause
#   - "If the answer is not in the context, say so" → explicit permission
#     to refuse; without this, LLMs often confabulate plausible answers.
#   - "Cite sources inline" → makes answers auditable by the end user.
#   - "concise, structured, markdown" → style control; prevents 500-word
#     essays when a bulleted list is better.
SYSTEM_PROMPT = """You are DocuMind, a precise and helpful assistant that answers \
questions strictly using the provided context from the user's documents.

Rules:
- Ground every answer in the provided context.
- If the answer is not in the context, say so clearly. Do not invent facts.
- Cite sources inline using [filename, page N] where available.
- Be concise, structured, and use markdown for lists and code.
"""

# The user prompt carries the retrieved context and the question. The
# `---` fences and explicit "Context" label help the LLM cleanly separate
# retrieved material from instructions — important both for answer quality
# and as a mild defense against prompt injection in uploaded documents.
USER_PROMPT = """Context from the user's documents:
---
{context}
---

Question: {question}

Answer using only the context above. Include citations like [filename, page N].
"""


@dataclass
class RagAnswer:
    """A simple container for one completed RAG query.

    We separate `answer` (the LLM text) from `sources` (the retrieved
    chunks) so the UI can render them in different sections — answer on
    top, sources in an expandable panel below.
    """

    question: str
    answer: str
    sources: list[Document]


def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks with a source header above each one.

    Why this shape? Putting a clear `[filename.pdf, page 3]` label directly
    above each chunk's text means when the LLM writes its answer, it can
    literally copy that label into its citation. No JSON output, no
    post-processing, no parsing — the cheapest reliable citation trick in
    RAG. It works because LLMs are very good at copying recent context.

    Note: PyPDFLoader's `page` metadata is 0-indexed, so we +1 to show
    human-readable page numbers.
    """
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        # Multiple fallbacks because different loaders populate different fields.
        filename = meta.get("filename") or meta.get("source") or f"source_{idx}"
        page = meta.get("page")
        header = f"[{filename}" + (f", page {page + 1}" if page is not None else "") + "]"
        blocks.append(f"{header}\n{doc.page_content}")
    # Blank line between chunks makes it visually obvious to the LLM where
    # one source ends and another begins.
    return "\n\n".join(blocks)


def _build_llm(model: str | None = None, *, streaming: bool = False) -> ChatOpenAI:
    """Construct a ChatOpenAI instance for either one-shot or streaming use.

    Why temperature 0.1 (not 0):
      - 0 is deterministic but sometimes produces repetitive artifacts.
      - 0.1 is effectively deterministic for factual questions but adds
        just enough variability to avoid those artifacts.
      - For RAG you never want creative temperatures (0.7+); the LLM's
        job is to summarize retrieved evidence, not invent things.

    The `streaming` flag tells OpenAI to use Server-Sent Events and push
    tokens as they're generated, rather than buffering the full response.
    See stream_answer() below for the consumer side.
    """
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        streaming=streaming,
    )


def retrieve(question: str, *, top_k: int | None = None) -> list[Document]:
    """Embed the question and return the K most similar chunks.

    What Chroma does under the hood:
      1. Calls OpenAIEmbeddings.embed_query(question) → 1536-dim vector.
      2. Walks the HNSW graph index to find approximate nearest neighbors.
      3. Returns the top-K closest chunks with their metadata.

    All typically < 50ms even with 100K chunks. See LEARN/03 for HNSW.
    """
    store = get_vector_store()
    k = top_k or settings.top_k
    # similarity_search() uses cosine similarity by default. Because OpenAI
    # embeddings are normalized, cosine == dot product == Euclidean-based
    # ranking — they all give the same ordering.
    return store.similarity_search(question, k=k)


def answer(question: str, *, model: str | None = None, top_k: int | None = None) -> RagAnswer:
    """Run a one-shot RAG query. Blocks until the full answer is ready.

    Flow (each step corresponds to a stage in RAG):
      1. RETRIEVE  — top-K chunks from the vector store.
      2. AUGMENT   — slot those chunks into the prompt template.
      3. GENERATE  — send to LLM, collect the full response.

    The LCEL chain `prompt | llm` is the composition. The `|` operator
    wires two Runnables together: the prompt's output (a formatted
    ChatPromptValue) becomes the llm's input. Every LCEL component supports
    the same `.invoke / .stream / .batch / .ainvoke` interface — that's
    what makes this ecosystem so compositional.
    """
    docs = retrieve(question, top_k=top_k)

    # ChatPromptTemplate.from_messages takes a list of (role, template) tuples
    # and substitutes {variables} when you call .invoke on the chain.
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    llm = _build_llm(model=model)

    # The pipe: prompt formats the messages, then llm calls the API.
    chain = prompt | llm

    # `invoke` is synchronous: wait for the full response, return it.
    response = chain.invoke(
        {"context": _format_context(docs), "question": question}
    )

    # response is an AIMessage; .content is the actual text.
    return RagAnswer(question=question, answer=response.content, sources=docs)


def stream_answer(
    question: str, *, model: str | None = None, top_k: int | None = None
) -> tuple[Iterator[str], list[Document]]:
    """Streaming version of answer().

    Returns two things so the UI can render in two phases:
      1. A generator that yields tokens as they arrive (drive the "typing"
         animation in the chat bubble).
      2. The retrieved sources, known upfront since retrieval is synchronous.
         These get displayed after the stream finishes.

    This two-phase shape is the same one used by ChatGPT's browsing mode
    and Perplexity: answer streams, sources drop in below when done.

    Why streaming matters: perceived latency. A 5-second response that
    starts showing text at 500ms feels dramatically faster than a 3-second
    response that shows nothing for 3 seconds. Non-negotiable for LLM UX.
    See LEARN/05-streaming-ui.md §1 for more.
    """
    # Retrieval is not streamed — it's a fast single RPC to Chroma.
    docs = retrieve(question, top_k=top_k)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    # streaming=True is the key difference from answer().
    llm = _build_llm(model=model, streaming=True)
    chain = prompt | llm

    def _stream() -> Iterator[str]:
        """Generator wrapping chain.stream() and yielding just the text content.

        chain.stream() yields AIMessageChunk objects — small incremental
        pieces of the response. Each chunk has .content (the text) plus
        metadata like token usage in the final chunk. We only want the
        text, so we yield chunk.content and filter out empty chunks (the
        first and last chunks from OpenAI are often empty metadata-only).
        """
        for chunk in chain.stream(
            {"context": _format_context(docs), "question": question}
        ):
            if chunk.content:
                yield chunk.content

    # Return the generator (NOT the exhausted result) plus the sources.
    return _stream(), docs
