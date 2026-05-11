"""
src/llm/gemini_client.py
========================
Google Gemini wrapper with automatic key rotation on quota errors.
"""
from __future__ import annotations

from typing import Generator, List

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.core import APIKeyManager, get_logger, is_quota_error

log = get_logger(__name__)


class RotatingChatLLM:
    def __init__(self, key_manager: APIKeyManager, model: str | None = None,
                 temperature: float | None = None, max_output_tokens: int | None = None):
        self.km = key_manager
        self.model_name = model or settings.gemini_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_output_tokens = max_output_tokens or settings.llm_max_output_tokens
        self._cache: dict[str, ChatGoogleGenerativeAI] = {}

    def _client(self, key: str) -> ChatGoogleGenerativeAI:
        if key not in self._cache:
            self._cache[key] = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=key,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        return self._cache[key]

    def invoke(self, messages: List[BaseMessage] | str):
        last_exc: Exception | None = None
        for _ in range(self.km.total or 1):
            key = self.km.get_active_key()
            try:
                return self._client(key).invoke(messages)
            except Exception as e:
                last_exc = e
                if is_quota_error(e):
                    log.warning("LLM quota error; rotating. (%s)", e)
                    self.km.mark_exhausted(key)
                    continue
                raise
        raise RuntimeError(f"All LLM keys exhausted: {last_exc}")

    def stream(self, messages: List[BaseMessage] | str) -> Generator:
        """Yield content chunks. Rotates only if the *first* call fails."""
        last_exc: Exception | None = None
        for _ in range(self.km.total or 1):
            key = self.km.get_active_key()
            yielded_any = False
            try:
                for chunk in self._client(key).stream(messages):
                    yielded_any = True
                    yield chunk
                return
            except Exception as e:
                last_exc = e
                if not yielded_any and is_quota_error(e):
                    log.warning("LLM stream quota error; rotating. (%s)", e)
                    self.km.mark_exhausted(key)
                    continue
                raise
        raise RuntimeError(f"All LLM keys exhausted (stream): {last_exc}")
