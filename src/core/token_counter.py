"""src/core/token_counter.py — tiktoken-based token counting."""
from __future__ import annotations
from typing import Iterable, Dict
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    try:
        return len(_ENCODING.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def count_tokens_for_messages(messages: Iterable[Dict]) -> int:
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", "")) + 2
    return total
