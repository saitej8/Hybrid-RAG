"""src/retrieval/chunker.py — Recursive chunking, optimized for retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkingResult:
    chunks: List[Document]
    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_count: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.chunk_count = len(self.chunks)


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> ChunkingResult:
    """Recursive splitter — works well for both Q&A and classification."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return ChunkingResult(splitter.split_documents(docs), chunk_size, chunk_overlap)
