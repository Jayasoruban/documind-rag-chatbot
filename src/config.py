"""
Application configuration loaded from environment variables.

Why this file exists
--------------------
Every tunable knob in DocuMind (which LLM? which embedder? chunk size?
where to store the vector DB?) is exposed as an environment variable.
Code reads them through a single immutable `Settings` object.

This is called "12-factor config". It gives us three big wins:

1. Redeployability — the same code runs in dev, staging, prod, and on
   a client's own servers just by swapping a `.env` file.
2. Safety — secrets (the OpenAI API key) never live in code or in git.
3. Clarity — every configurable value is listed in ONE place and has a
   sensible default.

See LEARN/06-production-next.md §1 for the full explanation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Reads the .env file (if present) and merges it into os.environ.
# In production (Docker, Streamlit Cloud) there's no .env file —
# env vars are injected by the platform and this call is a harmless no-op.
load_dotenv()

# Absolute path to the project root (the folder containing pyproject.toml).
# We resolve it once so other modules can build paths relative to the project
# instead of relative to the current working directory (which is fragile).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)  # frozen=True → Settings is immutable after creation.
class Settings:
    """Typed snapshot of all runtime configuration.

    We use a dataclass instead of raw os.getenv() calls scattered around
    the codebase because:
      - IDE autocomplete works (`settings.chunk_size` vs os.getenv strings)
      - Type hints catch "did you mean str or int?" bugs at dev time
      - One place to change when a new config option is added
    """

    openai_api_key: str
    chat_model: str           # Which OpenAI chat model to use (gpt-4o-mini default)
    embedding_model: str      # Which embedding model — MUST match at ingest & query time
    chunk_size: int           # Characters per chunk (see LEARN/02 for the trade-off)
    chunk_overlap: int        # Overlap between consecutive chunks (prevents cross-boundary loss)
    top_k: int                # How many chunks to retrieve per question
    chroma_persist_dir: Path  # Where ChromaDB writes its sqlite + HNSW files
    collection_name: str      # Namespace inside ChromaDB (enables multi-tenant later)

    @classmethod
    def load(cls) -> "Settings":
        """Read every knob from the environment and return a validated Settings.

        Called once at import time (see the `settings = Settings.load()` line
        at the bottom of this module). After that the object is frozen and
        shared across the whole app.
        """
        # .strip() defends against trailing whitespace or newlines — a very
        # common source of "invalid API key" errors when users copy-paste.
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        return cls(
            openai_api_key=api_key,
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            # Env vars are always strings — we convert to int at the boundary
            # so the rest of the codebase sees proper types.
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "4")),
            # Resolve the persist dir relative to the project root so it works
            # regardless of where streamlit is launched from.
            chroma_persist_dir=(
                PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "./chroma_db").lstrip("./")
            ),
            collection_name=os.getenv("COLLECTION_NAME", "documind"),
        )

    @property
    def has_api_key(self) -> bool:
        """True iff the user set a real-looking OpenAI key.

        Used by app.py to show a helpful error in the sidebar BEFORE any
        OpenAI call is made, instead of letting the app crash mid-query.
        Valid keys start with 'sk-' (or 'sk-proj-' for project keys).
        """
        return bool(self.openai_api_key) and self.openai_api_key.startswith("sk-")


# Module-level singleton. Every other file does `from .config import settings`
# and reads values off this object. Because it's frozen, there's no risk of
# accidental mutation.
settings = Settings.load()
