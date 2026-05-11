"""
src/retrieval/hybrid_retriever.py
=================================
Hybrid retrieval combining:
  * TF-IDF keyword search (scikit-learn)         — sparse, lexical
  * FAISS dense semantic search (Google embeds)  — semantic similarity

Scores are min-max normalized, then combined as:
    final = w_dense * dense_norm + w_sparse * sparse_norm

The top-K combined chunks are returned, with their per-method scores
exposed in metadata for transparency in the UI.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings
from src.core import get_logger
# FAISS distance for normalized embeddings is in [0, ~2]; map to [0, 1] similarity.
# Using exp decay handles wider distance ranges from newer embedding models.
import math
log = get_logger(__name__)


@dataclass
class ScoredDoc:
    document: Document
    dense_score: float
    sparse_score: float
    combined_score: float


class HybridRetriever:
    """Build once over a fixed chunk set, then `.search(query, k)` repeatedly."""

    def __init__(
        self,
        chunks: List[Document],
        embeddings,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ):
        if not chunks:
            raise ValueError("HybridRetriever requires at least one chunk.")

        self.chunks = chunks
        self.embeddings = embeddings
        self.dense_weight = dense_weight if dense_weight is not None else settings.hybrid_dense_weight
        self.sparse_weight = sparse_weight if sparse_weight is not None else settings.hybrid_sparse_weight

        log.info(
            "Building HybridRetriever (dense=%.2f, sparse=%.2f) over %d chunks",
            self.dense_weight, self.sparse_weight, len(chunks),
        )

        # ---- TF-IDF ------------------------------------------------------
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self.vectorizer.fit_transform(
            [c.page_content for c in chunks]
        )

        # ---- FAISS dense -------------------------------------------------
        self.vector_store: FAISS = FAISS.from_documents(chunks, embeddings)

    def search(self, query: str, k: int | None = None) -> List[ScoredDoc]:
        k = k or settings.retrieval_k
        if not query or not query.strip():
            return []

        pool_size = max(k * 4, 20)
        dense_hits: List[Tuple[Document, float]] = self.vector_store.similarity_search_with_score(
            query, k=min(pool_size, len(self.chunks))
        )
        dense_score_by_idx: dict[int, float] = {}
        for doc, dist in dense_hits:
            try:
                idx = next(
                    i for i, c in enumerate(self.chunks)
                    if c.page_content == doc.page_content
                    and c.metadata.get("page") == doc.metadata.get("page")
                )
            except StopIteration:
                continue
            sim = math.exp(-float(dist) / 2.0)
            
            dense_score_by_idx[idx] = sim

        q_vec = self.vectorizer.transform([query])
        sparse_arr = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        sparse_top_idx = np.argsort(-sparse_arr)[:pool_size]
        sparse_score_by_idx: dict[int, float] = {
            int(i): float(sparse_arr[i]) for i in sparse_top_idx if sparse_arr[i] > 0
        }

        candidates = set(dense_score_by_idx.keys()) | set(sparse_score_by_idx.keys())
        if not candidates:
            return []

        def _norm(scores: dict[int, float]) -> dict[int, float]:
            if not scores:
                return {}
            vals = np.array(list(scores.values()), dtype=np.float32)
            lo, hi = float(vals.min()), float(vals.max())
            rng = (hi - lo) or 1e-9
            return {i: (s - lo) / rng for i, s in scores.items()}

        dense_n = _norm(dense_score_by_idx)
        sparse_n = _norm(sparse_score_by_idx)

        scored: List[ScoredDoc] = []
        for idx in candidates:
            d = dense_n.get(idx, 0.0)
            s = sparse_n.get(idx, 0.0)
            combined = self.dense_weight * d + self.sparse_weight * s
            scored.append(ScoredDoc(
                document=self.chunks[idx],
                dense_score=round(dense_score_by_idx.get(idx, 0.0), 4),
                sparse_score=round(sparse_score_by_idx.get(idx, 0.0), 4),
                combined_score=round(combined, 4),
            ))

        scored.sort(key=lambda x: x.combined_score, reverse=True)
        return scored[:k]

    def save(self, chat_id: str) -> None:
        d = settings.vector_store_dir / chat_id
        d.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(d))
        with open(d / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(d / "tfidf.pkl", "wb") as f:
            pickle.dump((self.vectorizer, self._tfidf_matrix), f)
        log.info("Saved HybridRetriever for chat %s", chat_id[:8])

    @classmethod
    def load(cls, chat_id: str, embeddings) -> Optional["HybridRetriever"]:
        d = settings.vector_store_dir / chat_id
        if not (d / "index.faiss").exists():
            return None
        try:
            vs = FAISS.load_local(str(d), embeddings, allow_dangerous_deserialization=True)
            with open(d / "chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
            obj = cls.__new__(cls)
            obj.chunks = chunks
            obj.embeddings = embeddings
            obj.dense_weight = settings.hybrid_dense_weight
            obj.sparse_weight = settings.hybrid_sparse_weight
            obj.vector_store = vs
            with open(d / "tfidf.pkl", "rb") as f:
                obj.vectorizer, obj._tfidf_matrix = pickle.load(f)
            log.info("Restored HybridRetriever for chat %s (%d chunks)", chat_id[:8], len(chunks))
            return obj
        except Exception as e:
            log.error("Failed to restore HybridRetriever for %s: %s", chat_id[:8], e)
            return None


def delete_vector_store(chat_id: str) -> None:
    d = settings.vector_store_dir / chat_id
    if not d.exists():
        return
    for p in d.iterdir():
        try: p.unlink()
        except Exception: pass
    try: d.rmdir()
    except Exception: pass
