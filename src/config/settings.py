"""
src/config/settings.py
======================
Centralised, typed app config.
Reads from .env locally and from st.secrets on Streamlit Cloud.
Supports comma-separated GOOGLE_API_KEYS for automatic key rotation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_keys(raw: str) -> List[str]:
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


class Settings(BaseSettings):
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_api_keys_raw: str = Field(default="", alias="GOOGLE_API_KEYS")

    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    embedding_model: str = Field(default="models/embedding-001", alias="EMBEDDING_MODEL")

    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_output_tokens: int = Field(default=2048, alias="LLM_MAX_OUTPUT_TOKENS")

    retrieval_k: int = Field(default=5, alias="RETRIEVAL_K")
    hybrid_dense_weight: float = Field(default=0.6, alias="HYBRID_DENSE_WEIGHT")
    hybrid_sparse_weight: float = Field(default=0.4, alias="HYBRID_SPARSE_WEIGHT")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    chat_db_path: Path = Field(default=Path("data/chat_memory.db"), alias="CHAT_DB_PATH")
    vector_store_dir: Path = Field(
        default=Path("data/vector_stores"), alias="VECTOR_STORE_DIR"
    )

    max_pdf_mb: int = Field(default=25, alias="MAX_PDF_MB")
    max_input_chars: int = Field(default=4000, alias="MAX_INPUT_CHARS")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # Set LOG_FILE="" in secrets/env to disable file logging (recommended on
    # Streamlit Cloud where the filesystem is ephemeral).
    log_file: str = Field(default="data/logs/app.log", alias="LOG_FILE")

    enable_streaming: bool = Field(default=True, alias="ENABLE_STREAMING")
    enable_evaluation: bool = Field(default=True, alias="ENABLE_EVALUATION")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def all_default_keys(self) -> List[str]:
        """All keys from .env / secrets (single + multi), de-duplicated, ordered."""
        keys: List[str] = []
        if self.google_api_key:
            keys.append(self.google_api_key.strip())
        keys.extend(_split_keys(self.google_api_keys_raw))
        seen, out = set(), []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    @property
    def log_file_path(self) -> Optional[Path]:
        """Returns the resolved log file path, or None if file logging is disabled."""
        s = (self.log_file or "").strip()
        return Path(s) if s else None

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chat_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        lf = self.log_file_path
        if lf is not None:
            try:
                lf.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass


def _load_streamlit_secrets_into_env() -> None:
    """If running on Streamlit Cloud, copy st.secrets values into os.environ
    so pydantic-settings picks them up. Safe no-op locally."""
    try:
        import os
        import streamlit as st
        if not hasattr(st, "secrets"):
            return
        for k in list(st.secrets.keys()):
            v = st.secrets[k]
            if isinstance(v, (str, int, float, bool)) and k not in os.environ:
                os.environ[k] = str(v)
    except Exception:
        pass


_load_streamlit_secrets_into_env()
settings = Settings()
settings.ensure_dirs()
