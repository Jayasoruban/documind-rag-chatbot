"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chroma_persist_dir: Path
    collection_name: str

    @classmethod
    def load(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        return cls(
            openai_api_key=api_key,
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "4")),
            chroma_persist_dir=(
                PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "./chroma_db").lstrip("./")
            ),
            collection_name=os.getenv("COLLECTION_NAME", "documind"),
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key) and self.openai_api_key.startswith("sk-")


settings = Settings.load()
