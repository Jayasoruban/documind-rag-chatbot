"""
DocuMind — Streamlit chat UI for the RAG pipeline in src/.

This file is the ONLY non-replaceable piece. Everything in src/ (loader,
vector_store, rag_chain) is UI-agnostic: tomorrow we could swap this file
for a FastAPI + Next.js frontend or a Slack bot with zero changes to src/.
That clean separation is why clients respect layered projects.

Quick mental model of how Streamlit runs this file:
  - The whole script re-executes top-to-bottom on every user interaction
    (button click, text input, file upload, model-dropdown change).
  - State that must survive a re-run lives in `st.session_state`, a dict
    tied to the user's browser session.
  - UI is declared, not managed: you write `st.markdown(...)` in order
    and Streamlit renders in that order.

See LEARN/05-streaming-ui.md for the deep dive on session state, streaming,
and Streamlit's mental model.
"""

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


# set_page_config MUST be the first Streamlit call — Streamlit enforces this
# or it errors. Sets the browser tab title, icon, default layout, etc.
st.set_page_config(
    page_title="DocuMind — Chat with your documents",
    page_icon="📚",
    layout="wide",                        # use full browser width; default is narrow
    initial_sidebar_state="expanded",     # sidebar open on first load
)


def _init_state() -> None:
    """Seed session state with default values on the first run only.

    The `if "key" not in session_state` idiom is the standard Streamlit
    pattern for initializing state. On run 1 the keys don't exist and get
    created; on every subsequent re-run the if-condition is False so
    existing values (like the chat history) survive.
    """
    if "messages" not in st.session_state:
        # Each message is a dict: {"role": "user" | "assistant", "content": str, "sources"?: [...]}
        # Keeping a simple list makes it trivial to re-render on each run.
        st.session_state.messages = []
    if "model" not in st.session_state:
        # User-selectable via the sidebar dropdown; default comes from .env
        st.session_state.model = settings.chat_model


def _render_sidebar() -> None:
    """Build the left sidebar: branding, API-key guard, model picker,
    upload section, and knowledge-base manager."""
    # `with st.sidebar:` tells every st.* call inside this block to render
    # into the sidebar column instead of the main page.
    with st.sidebar:
        st.markdown("## 📚 DocuMind")
        st.caption("Chat with your documents. Private, cited, production-ready.")

        st.divider()

        # Fail-fast guard. We'd rather show a clear error and halt than
        # let the user upload docs and then have the first query fail with
        # a confusing OpenAI authentication error deep in the stack.
        if not settings.has_api_key:
            st.error("OPENAI_API_KEY is missing. Set it in `.env` and restart.")
            # st.stop() halts this run. The user sees what's rendered so
            # far; nothing below this line executes.
            st.stop()

        # Model picker. We expose four OpenAI models — mini and full in
        # both 4o and 4.1 families. Users can bump to a bigger model for
        # hard questions at ~10x the cost per query.
        # The returned value is assigned back to session_state so it
        # persists across runs; without that, the dropdown would reset.
        st.session_state.model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
            index=0,
            help="gpt-4o-mini is the default. Switch up for harder questions.",
        )

        st.divider()
        st.markdown("### 📥 Upload documents")

        # file_uploader returns None until the user drops files; then it
        # returns a list of UploadedFile objects (file-like objects in memory).
        # `type=...` filters the OS file picker to only matching extensions;
        # accept_multiple_files=True enables multi-select.
        uploaded = st.file_uploader(
            "PDF, TXT, MD, or DOCX",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        # The button is the trigger. We don't ingest on upload because the
        # user might want to add several files first; requiring a click lets
        # them batch the indexing operation.
        if uploaded and st.button("Index documents", type="primary", use_container_width=True):
            _index_uploads(uploaded)

        st.divider()
        _render_knowledge_base()

        st.divider()
        # st.expander = collapsible section. Keeps the sidebar tidy while
        # still giving recruiters/visitors context without leaving the app.
        with st.expander("About this project"):
            st.markdown(
                "**DocuMind** is a retrieval-augmented chatbot over your documents.\n\n"
                "**Stack:** Python · LangChain · OpenAI · ChromaDB · Streamlit\n\n"
                "Built by [Jayasoruban R](https://github.com/jayasoruban)."
            )


def _index_uploads(uploaded_files) -> None:
    """Take uploaded in-memory files, persist them to disk temporarily so
    LangChain loaders can read them, then run the full ingest pipeline:
    load → chunk → embed → store.

    This function is where all three concepts from LEARN/02 come together.
    """
    # st.spinner() shows a loading spinner with a message until the `with`
    # block exits. Cheap, effective UX signal.
    with st.spinner(f"Indexing {len(uploaded_files)} file(s)…"):
        # tempfile.TemporaryDirectory is a context manager that creates a
        # directory and automatically deletes it (and everything inside)
        # when the block exits. The uploaded bytes NEVER persist on disk
        # beyond this function — important for privacy.
        with tempfile.TemporaryDirectory() as tmp:
            paths: list[Path] = []
            for file in uploaded_files:
                # file.getbuffer() returns raw bytes. Write them to a file
                # inside the temp dir so LangChain loaders (which need a
                # path, not bytes) can read them.
                dest = Path(tmp) / file.name
                dest.write_bytes(file.getbuffer())
                paths.append(dest)

            # The classic ingest pipeline:
            docs = load_files(paths)                       # bytes → Documents
            chunks = chunk_documents(                      # Documents → small overlapping chunks
                docs,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            added = add_documents(chunks)                  # chunks → embedded vectors in Chroma

    st.success(f"Indexed {len(uploaded_files)} file(s) into {added} chunks.")
    # Force an immediate re-run so the sidebar's knowledge-base list shows
    # the newly indexed files without waiting for another interaction.
    st.rerun()


def _render_knowledge_base() -> None:
    """Show the list of currently indexed files with per-file delete buttons
    and a 'nuke everything' reset button."""
    st.markdown("### 🗂️ Knowledge base")
    sources = list_sources()
    total_chunks = count_chunks()

    # Empty state. Always handle the "no data yet" case — it's the first
    # thing every new user sees.
    if not sources:
        st.caption("No documents yet. Upload something to start.")
        return

    st.caption(f"{len(sources)} file(s) · {total_chunks} chunks")

    for source in sources:
        # st.columns([4, 1]) creates two side-by-side columns with width
        # ratio 4:1 — filename takes most of the space, delete button is tiny.
        col_name, col_btn = st.columns([4, 1])
        col_name.markdown(f"• `{source}`")

        # key=... is REQUIRED because Streamlit deduplicates widgets by key.
        # Without a unique key per iteration, the loop would create N
        # buttons with the same identity and Streamlit would error out.
        if col_btn.button("✕", key=f"del_{source}", help=f"Remove {source}"):
            removed = delete_by_source(source)
            # st.toast shows a short floating notification; non-blocking,
            # nicer UX than st.success for quick confirmations.
            st.toast(f"Removed {removed} chunk(s) from {source}")
            st.rerun()

    if st.button("Clear entire knowledge base", use_container_width=True):
        reset_store()
        st.toast("Knowledge base cleared.")
        st.rerun()


def _render_hero() -> None:
    """Top-of-page headline. Pure cosmetic — sets tone and explains the app
    to first-time visitors in one glance."""
    st.markdown("# 📚 DocuMind")
    st.markdown(
        "#### Chat with your documents. Get cited answers. Keep everything private."
    )
    st.caption(
        "Upload PDFs, text files, markdown, or Word documents in the sidebar → "
        "ask questions below."
    )


def _render_chat() -> None:
    """The main chat UI: render history, accept a new question, stream the
    answer, and append everything to session_state for the next run."""

    # Re-render the full chat history on every run. st.chat_message(role)
    # is a context manager that wraps its content in a styled chat bubble
    # (user bubble vs assistant bubble are visually distinct by role).
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Assistant messages carry a `sources` list; render it below
            # their content so users can verify where the answer came from.
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # st.chat_input is a text input pinned to the bottom of the page.
    # Returns None on most runs; returns the submitted string when the
    # user hits Enter. That's why we early-return on falsy below.
    prompt = st.chat_input("Ask anything about your documents…")
    if not prompt:
        return

    # Guard against asking questions against an empty knowledge base.
    if count_chunks() == 0:
        st.warning("Upload and index a document first.")
        return

    # Persist and render the user's turn BEFORE starting the LLM call so
    # they see their message immediately (even during the 500ms of
    # retrieval latency before tokens start streaming).
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # The assistant's turn — where streaming happens.
    with st.chat_message("assistant"):
        # st.empty() creates a single placeholder slot. Subsequent calls
        # to placeholder.markdown() REPLACE its contents (not append).
        # That's the trick that lets us build the "live typing" effect
        # by repeatedly overwriting with a growing buffer.
        placeholder = st.empty()

        try:
            # stream_answer returns (generator, sources). Sources are known
            # up front; the generator yields tokens as OpenAI emits them.
            token_iter, sources = stream_answer(prompt, model=st.session_state.model)

            buffer = ""
            for token in token_iter:
                buffer += token
                # The "▌" suffix is a fake blinking-cursor character that
                # gives the UI a typing feel while tokens stream.
                placeholder.markdown(buffer + "▌")
            # After streaming completes, overwrite one last time WITHOUT
            # the cursor to give a clean final render.
            placeholder.markdown(buffer)
        except Exception as exc:
            # Catch ANY exception from the LLM call (rate limits, auth
            # failures, network blips) and show it in-chat instead of
            # letting the whole Streamlit app crash. The user can just ask
            # another question.
            buffer = f"Something went wrong: `{exc}`"
            placeholder.error(buffer)
            sources = []

        if sources:
            _render_sources(sources)

    # Persist the assistant's turn for future re-runs. Note we serialize
    # Document objects to plain dicts via _serialize_source() — Documents
    # don't round-trip cleanly through session state, so we store the only
    # fields the UI actually needs.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": buffer,
            "sources": [_serialize_source(d) for d in sources],
        }
    )


def _serialize_source(doc) -> dict:
    """Flatten a LangChain Document into a plain dict for session_state.

    We only keep the fields the sources panel needs to render. Excerpt is
    truncated to 500 chars to keep session state small and the UI snappy.
    """
    return {
        "filename": doc.metadata.get("filename") or doc.metadata.get("source", "?"),
        "page": doc.metadata.get("page"),
        "excerpt": doc.page_content[:500],
    }


def _render_sources(sources) -> None:
    """Render the expandable 'Sources' panel below an assistant message.

    This function has to handle BOTH shapes:
      - Live Document objects (fresh from stream_answer, before persisting)
      - Plain dicts (rehydrated from session_state on re-runs)

    The hasattr(src, 'metadata') check discriminates between them.
    """
    # st.expander is collapsed by default. Label includes the count so
    # users know how many sources were used without opening.
    with st.expander(f"🔎 Sources ({len(sources)})"):
        for idx, src in enumerate(sources, start=1):
            if hasattr(src, "metadata"):
                # Fresh LangChain Document — pull fields from its .metadata.
                filename = src.metadata.get("filename") or src.metadata.get("source", "?")
                page = src.metadata.get("page")
                excerpt = src.page_content[:500]
            else:
                # Rehydrated dict shape from session_state.
                filename = src["filename"]
                page = src.get("page")
                excerpt = src["excerpt"]

            # PyPDFLoader's page is 0-indexed; +1 for human display.
            page_label = f" — page {page + 1}" if page is not None else ""
            st.markdown(f"**{idx}. `{filename}`{page_label}**")
            # A little inline CSS for the excerpt block — subtle gray
            # background + rounded corners, consistent look across themes.
            # unsafe_allow_html=True is required whenever we pass raw HTML.
            st.markdown(
                f"<div style='padding:8px 12px;background:#f6f7f9;border-radius:6px;"
                f"color:#333;font-size:0.88rem;'>{excerpt}…</div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    """Entry point — called by the `if __name__ == "__main__":` guard.

    Order matters here because Streamlit renders top-to-bottom:
      1. Initialize state first (so everything else can read it).
      2. Sidebar (renders left column independently).
      3. Hero (main column, top of page).
      4. Chat (main column, scrollable message history + pinned input).
    """
    _init_state()
    _render_sidebar()
    _render_hero()
    _render_chat()


if __name__ == "__main__":
    main()
