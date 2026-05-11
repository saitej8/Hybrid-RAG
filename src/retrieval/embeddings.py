"""
src/retrieval/embeddings.py
===========================
Google embeddings wrapper with API-key rotation.
"""
from __future__ import annotations

from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings
from src.core import APIKeyManager, get_logger, is_quota_error

log = get_logger(__name__)


class RotatingEmbeddings:
    """Drop-in `embed_documents` / `embed_query` with key rotation."""

    def __init__(self, key_manager: APIKeyManager, model: str | None = None):
        self.km = key_manager
        self.model_name = model or settings.embedding_model
        self._client_for_key: dict[str, GoogleGenerativeAIEmbeddings] = {}

    def _client(self, key: str) -> GoogleGenerativeAIEmbeddings:
        if key not in self._client_for_key:
            self._client_for_key[key] = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=key,
            )
        return self._client_for_key[key]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        last_exc: Exception | None = None
        for _ in range(self.km.total or 1):
            key = self.km.get_active_key()
            try:
                return self._client(key).embed_documents(texts)
            except Exception as e:
                last_exc = e
                if is_quota_error(e):
                    log.warning("Embed quota error; rotating key. (%s)", e)
                    self.km.mark_exhausted(key)
                    continue
                raise
        raise RuntimeError(f"All keys exhausted while embedding documents: {last_exc}")

    def embed_query(self, text: str) -> List[float]:
        last_exc: Exception | None = None
        for _ in range(self.km.total or 1):
            key = self.km.get_active_key()
            try:
                return self._client(key).embed_query(text)
            except Exception as e:
                last_exc = e
                if is_quota_error(e):
                    log.warning("Embed-query quota; rotating key. (%s)", e)
                    self.km.mark_exhausted(key)
                    continue
                raise
        raise RuntimeError(f"All keys exhausted while embedding query: {last_exc}")

    # ⭐ Some LangChain code paths invoke embeddings as a callable.
    def __call__(self, text: str):
        return self.embed_query(text)
