"""
src/core/key_manager.py
=======================
Google API key pool with automatic rotation on quota exhaustion.

How keys are sourced (priority order):
    1. UI-pasted key (highest)
    2. .env / Streamlit secrets — GOOGLE_API_KEY (single)
    3. .env / Streamlit secrets — GOOGLE_API_KEYS (comma-separated)

Rotation logic:
    * The active key is whichever one we're currently using.
    * On a quota / 429 / RESOURCE_EXHAUSTED error, the key is marked
      "exhausted" for `cooldown_seconds` (default 60s) and we move to
      the next available key.
    * If all keys are exhausted, the manager raises NoKeysAvailable.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

from src.core.logger import get_logger

log = get_logger(__name__)


class NoKeysAvailable(RuntimeError):
    pass


def is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    indicators = (
        "429", "quota", "rate limit", "rate-limit",
        "resource has been exhausted", "resource_exhausted",
        "exhausted", "too many requests",
    )
    return any(k in msg for k in indicators)


class APIKeyManager:
    def __init__(self, keys: List[str], cooldown_seconds: int = 60):
        seen, ordered = set(), []
        for k in keys:
            k = (k or "").strip()
            if k and k not in seen:
                seen.add(k); ordered.append(k)

        self._keys: List[str] = ordered
        self._cooldown = cooldown_seconds
        self._exhausted_until: dict[str, float] = {}
        self._idx: int = 0
        self._lock = threading.Lock()

        log.info("APIKeyManager initialized with %d key(s)", len(self._keys))

    @property
    def total(self) -> int:
        return len(self._keys)

    def has_keys(self) -> bool:
        return bool(self._keys)

    def all_keys_masked(self) -> List[str]:
        return [self._mask(k) for k in self._keys]

    def get_active_key(self) -> str:
        with self._lock:
            if not self._keys:
                raise NoKeysAvailable("No Google API keys configured.")
            now = time.time()
            for _ in range(len(self._keys)):
                k = self._keys[self._idx]
                if self._exhausted_until.get(k, 0) <= now:
                    return k
                self._idx = (self._idx + 1) % len(self._keys)
            soonest = min(self._exhausted_until.items(), key=lambda kv: kv[1])
            wait = max(0, soonest[1] - now)
            raise NoKeysAvailable(
                f"All {len(self._keys)} API keys are rate-limited. "
                f"Try again in ~{int(wait)}s."
            )

    def mark_exhausted(self, key: str, cooldown: Optional[int] = None) -> None:
        with self._lock:
            wait = cooldown if cooldown is not None else self._cooldown
            self._exhausted_until[key] = time.time() + wait
            log.warning("Key %s marked exhausted for %ds (rotating)",
                        self._mask(key), wait)
            if key in self._keys:
                cur = self._keys.index(key)
                self._idx = (cur + 1) % len(self._keys)

    def replace_keys(self, keys: List[str]) -> None:
        with self._lock:
            seen, ordered = set(), []
            for k in keys:
                k = (k or "").strip()
                if k and k not in seen:
                    seen.add(k); ordered.append(k)
            self._keys = ordered
            self._exhausted_until.clear()
            self._idx = 0
        log.info("Key pool replaced — now %d key(s)", len(self._keys))

    @staticmethod
    def _mask(k: str) -> str:
        if not k:
            return "(empty)"
        if len(k) <= 8:
            return "*" * len(k)
        return f"{k[:4]}…{k[-4:]}"
