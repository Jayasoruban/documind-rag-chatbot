"""Retrieval-Augmented Generation pipeline built on LangChain LCEL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import settings
from .vector_store import get_vector_store

SYSTEM_PROMPT = """You are DocuMind, a precise and helpful assistant that answers \
questions strictly using the provided context from the user's documents.

Rules:
- Ground every answer in the provided context.
- If the answer is not in the context, say so clearly. Do not invent facts.
- Cite sources inline using [filename, page N] where available.
- Be concise, structured, and use markdown for lists and code.
"""

USER_PROMPT = """Context from the user's documents:
---
{context}
---

Question: {question}

Answer using only the context above. Include citations like [filename, page N].
"""


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[Document]


def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks with source labels the LLM can cite."""
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        filename = meta.get("filename") or meta.get("source") or f"source_{idx}"
        page = meta.get("page")
        header = f"[{filename}" + (f", page {page + 1}" if page is not None else "") + "]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _build_llm(model: str | None = None, *, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        streaming=streaming,
    )


def retrieve(question: str, *, top_k: int | None = None) -> list[Document]:
    """Fetch the most relevant chunks for the question."""
    store = get_vector_store()
    k = top_k or settings.top_k
    return store.similarity_search(question, k=k)


def answer(question: str, *, model: str | None = None, top_k: int | None = None) -> RagAnswer:
    """Run a one-shot RAG query and return the answer with its source chunks."""
    docs = retrieve(question, top_k=top_k)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    llm = _build_llm(model=model)
    chain = prompt | llm
    response = chain.invoke(
        {"context": _format_context(docs), "question": question}
    )
    return RagAnswer(question=question, answer=response.content, sources=docs)


def stream_answer(
    question: str, *, model: str | None = None, top_k: int | None = None
) -> tuple[Iterator[str], list[Document]]:
    """Stream tokens as they arrive. Returns (token_iterator, sources)."""
    docs = retrieve(question, top_k=top_k)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    llm = _build_llm(model=model, streaming=True)
    chain = prompt | llm

    def _stream() -> Iterator[str]:
        for chunk in chain.stream(
            {"context": _format_context(docs), "question": question}
        ):
            if chunk.content:
                yield chunk.content

    return _stream(), docs
