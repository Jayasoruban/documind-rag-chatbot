"""DocuMind — Streamlit chat UI for a local RAG knowledge base."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.config import settings
from src.document_loader import (
    SUPPORTED_EXTENSIONS,
    chunk_documents,
    load_files,
)
from src.rag_chain import stream_answer
from src.vector_store import (
    add_documents,
    count_chunks,
    delete_by_source,
    list_sources,
    reset_store,
)


st.set_page_config(
    page_title="DocuMind — Chat with your documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = settings.chat_model


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📚 DocuMind")
        st.caption("Chat with your documents. Private, cited, production-ready.")

        st.divider()

        if not settings.has_api_key:
            st.error("OPENAI_API_KEY is missing. Set it in `.env` and restart.")
            st.stop()

        st.session_state.model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
            index=0,
            help="gpt-4o-mini is the default. Switch up for harder questions.",
        )

        st.divider()
        st.markdown("### 📥 Upload documents")
        uploaded = st.file_uploader(
            "PDF, TXT, MD, or DOCX",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Index documents", type="primary", use_container_width=True):
            _index_uploads(uploaded)

        st.divider()
        _render_knowledge_base()

        st.divider()
        with st.expander("About this project"):
            st.markdown(
                "**DocuMind** is a retrieval-augmented chatbot over your documents.\n\n"
                "**Stack:** Python · LangChain · OpenAI · ChromaDB · Streamlit\n\n"
                "Built by [Jayasoruban R](https://github.com/jayasoruban)."
            )


def _index_uploads(uploaded_files) -> None:
    with st.spinner(f"Indexing {len(uploaded_files)} file(s)…"):
        with tempfile.TemporaryDirectory() as tmp:
            paths: list[Path] = []
            for file in uploaded_files:
                dest = Path(tmp) / file.name
                dest.write_bytes(file.getbuffer())
                paths.append(dest)
            docs = load_files(paths)
            chunks = chunk_documents(
                docs,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            added = add_documents(chunks)
    st.success(f"Indexed {len(uploaded_files)} file(s) into {added} chunks.")
    st.rerun()


def _render_knowledge_base() -> None:
    st.markdown("### 🗂️ Knowledge base")
    sources = list_sources()
    total_chunks = count_chunks()

    if not sources:
        st.caption("No documents yet. Upload something to start.")
        return

    st.caption(f"{len(sources)} file(s) · {total_chunks} chunks")
    for source in sources:
        col_name, col_btn = st.columns([4, 1])
        col_name.markdown(f"• `{source}`")
        if col_btn.button("✕", key=f"del_{source}", help=f"Remove {source}"):
            removed = delete_by_source(source)
            st.toast(f"Removed {removed} chunk(s) from {source}")
            st.rerun()

    if st.button("Clear entire knowledge base", use_container_width=True):
        reset_store()
        st.toast("Knowledge base cleared.")
        st.rerun()


def _render_hero() -> None:
    st.markdown("# 📚 DocuMind")
    st.markdown(
        "#### Chat with your documents. Get cited answers. Keep everything private."
    )
    st.caption(
        "Upload PDFs, text files, markdown, or Word documents in the sidebar → "
        "ask questions below."
    )


def _render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    prompt = st.chat_input("Ask anything about your documents…")
    if not prompt:
        return

    if count_chunks() == 0:
        st.warning("Upload and index a document first.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            token_iter, sources = stream_answer(prompt, model=st.session_state.model)
            buffer = ""
            for token in token_iter:
                buffer += token
                placeholder.markdown(buffer + "▌")
            placeholder.markdown(buffer)
        except Exception as exc:
            buffer = f"Something went wrong: `{exc}`"
            placeholder.error(buffer)
            sources = []

        if sources:
            _render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": buffer,
            "sources": [_serialize_source(d) for d in sources],
        }
    )


def _serialize_source(doc) -> dict:
    return {
        "filename": doc.metadata.get("filename") or doc.metadata.get("source", "?"),
        "page": doc.metadata.get("page"),
        "excerpt": doc.page_content[:500],
    }


def _render_sources(sources) -> None:
    with st.expander(f"🔎 Sources ({len(sources)})"):
        for idx, src in enumerate(sources, start=1):
            if hasattr(src, "metadata"):
                filename = src.metadata.get("filename") or src.metadata.get("source", "?")
                page = src.metadata.get("page")
                excerpt = src.page_content[:500]
            else:
                filename = src["filename"]
                page = src.get("page")
                excerpt = src["excerpt"]
            page_label = f" — page {page + 1}" if page is not None else ""
            st.markdown(f"**{idx}. `{filename}`{page_label}**")
            st.markdown(
                f"<div style='padding:8px 12px;background:#f6f7f9;border-radius:6px;"
                f"color:#333;font-size:0.88rem;'>{excerpt}…</div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    _init_state()
    _render_sidebar()
    _render_hero()
    _render_chat()


if __name__ == "__main__":
    main()
