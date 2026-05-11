"""
src/evaluation/ragas_evaluator.py
=================================
RAG evaluation per response.

Tries the real `ragas` library first (faithfulness, answer_relevancy,
context_precision). If ragas is not installed or its harness fails,
falls back to a lightweight LLM-judged evaluator that returns the same
shape so the UI never breaks.

All scores in [0, 1]; higher = better.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from langchain_core.documents import Document

from src.core import get_logger

log = get_logger(__name__)


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) or 1e-9
    return float(np.dot(a_arr, b_arr) / denom)


def _safe_json_extract(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class RAGMetrics:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall: float
    backend: str = "custom"

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def empty(cls) -> "RAGMetrics":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, backend="none")


class _CustomEvaluator:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    def _faithfulness(self, answer: str, contexts: List[str]) -> float:
        sentences = _split_sentences(answer)
        if not sentences or not contexts:
            return 0.0
        joined = "\n\n".join(contexts)[:6000]
        prompt = (
            "You are a strict fact-checker. Given the CONTEXT and CLAIMS, "
            "decide how many claims are entailed by the context.\n\n"
            f"CONTEXT:\n\"\"\"\n{joined}\n\"\"\"\n\n"
            f"CLAIMS:\n{json.dumps(sentences)}\n\n"
            'Reply ONLY: {"supported": <int>, "total": <int>}'
        )
        try:
            resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
            data = _safe_json_extract(text) or {}
            supported = int(data.get("supported", 0))
            total = int(data.get("total", len(sentences))) or len(sentences)
            return max(0.0, min(1.0, supported / total))
        except Exception:
            return self._lex_overlap(answer, joined)

    def _answer_relevancy(self, question: str, answer: str) -> float:
        if not question.strip() or not answer.strip():
            return 0.0
        prompt = (
            "Generate 3 different questions that the ANSWER below would "
            "correctly answer. Reply ONLY with a JSON array of 3 strings.\n\n"
            f"ANSWER:\n\"\"\"\n{answer[:3000]}\n\"\"\""
        )
        candidates: List[str] = []
        try:
            resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    candidates = [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            candidates = []

        if not candidates:
            try:
                q_emb = self.embeddings.embed_query(question)
                a_emb = self.embeddings.embed_query(answer[:2000])
                return max(0.0, _cosine(q_emb, a_emb))
            except Exception:
                return self._lex_overlap(question, answer)

        try:
            q_emb = self.embeddings.embed_query(question)
            cand_embs = self.embeddings.embed_documents(candidates)
            sims = [max(0.0, _cosine(q_emb, c)) for c in cand_embs]
            return float(sum(sims) / len(sims)) if sims else 0.0
        except Exception:
            return self._lex_overlap(question, " ".join(candidates))

    def _context_precision(self, question: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        ctx = [c[:1500] for c in contexts]
        prompt = (
            "For the QUESTION below, label each CONTEXT as RELEVANT (1) or "
            "NOT RELEVANT (0).\n\n"
            f"QUESTION: {question}\n\n"
            "CONTEXTS:\n"
            + "\n".join(f"[{i}] {c}" for i, c in enumerate(ctx))
            + '\n\nReply ONLY: {"labels": [0 or 1, ...]} in same order.'
        )
        try:
            resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
            data = _safe_json_extract(text) or {}
            labels = data.get("labels") or []
            if labels and len(labels) == len(contexts):
                return float(sum(int(bool(x)) for x in labels) / len(labels))
        except Exception:
            pass
        try:
            q_emb = self.embeddings.embed_query(question)
            embs = self.embeddings.embed_documents(ctx)
            sims = [_cosine(q_emb, e) for e in embs]
            relevant = sum(1 for s in sims if s >= 0.45)
            return float(relevant / len(sims)) if sims else 0.0
        except Exception:
            return 0.0

    def _context_recall(self, answer: str, contexts: List[str]) -> float:
        if not answer.strip() or not contexts:
            return 0.0
        sentences = _split_sentences(answer)
        if not sentences:
            return 0.0
        joined = "\n\n".join(contexts)[:6000]
        prompt = (
            "Given the CONTEXT, how many CLAIMS are covered by it?\n\n"
            f"CONTEXT:\n\"\"\"\n{joined}\n\"\"\"\n\n"
            f"CLAIMS:\n{json.dumps(sentences)}\n\n"
            'Reply ONLY: {"covered": <int>, "total": <int>}'
        )
        try:
            resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
            data = _safe_json_extract(text) or {}
            covered = int(data.get("covered", 0))
            total = int(data.get("total", len(sentences))) or len(sentences)
            return max(0.0, min(1.0, covered / total))
        except Exception:
            return self._lex_overlap(answer, joined)

    @staticmethod
    def _lex_overlap(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        at = set(re.findall(r"\w+", a.lower()))
        bt = set(re.findall(r"\w+", b.lower()))
        if not at:
            return 0.0
        return len(at & bt) / len(at)


class RAGEvaluator:
    def __init__(self, llm, embeddings, prefer_ragas: bool = True):
        self.llm = llm
        self.embeddings = embeddings
        self.prefer_ragas = prefer_ragas
        self._custom = _CustomEvaluator(llm, embeddings)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[Document] | List[str],
    ) -> RAGMetrics:
        ctx_strs: List[str] = []
        for c in contexts or []:
            ctx_strs.append(c.page_content if isinstance(c, Document) else str(c))

        if self.prefer_ragas:
            try:
                from ragas import evaluate as ragas_evaluate
                from ragas.metrics import (
                    answer_relevancy as r_relev,
                    context_precision as r_cprec,
                    faithfulness as r_faith,
                )
                from datasets import Dataset

                ds = Dataset.from_dict({
                    "question": [question],
                    "answer": [answer],
                    "contexts": [ctx_strs],
                    "ground_truth": [answer],
                })
                result = ragas_evaluate(
                    ds,
                    metrics=[r_faith, r_relev, r_cprec],
                    llm=self.llm,
                    embeddings=self.embeddings,
                    raise_exceptions=False,
                )
                df = result.to_pandas()

                def _g(col: str) -> float:
                    if col not in df.columns or df.empty:
                        return 0.0
                    val = df.iloc[0][col]
                    try:
                        v = float(val)
                        return 0.0 if (v != v) else max(0.0, min(1.0, v))
                    except Exception:
                        return 0.0

                faith = _g("faithfulness")
                relev = _g("answer_relevancy")
                cprec = _g("context_precision")
                crec = self._custom._context_recall(answer, ctx_strs)
                overall = round((faith + relev + cprec + crec) / 4.0, 3)
                return RAGMetrics(
                    faithfulness=round(faith, 3),
                    answer_relevancy=round(relev, 3),
                    context_precision=round(cprec, 3),
                    context_recall=round(crec, 3),
                    overall=overall,
                    backend="ragas",
                )
            except Exception as e:
                log.warning("ragas backend failed (%s) — using custom evaluator", e)

        try: faith = self._custom._faithfulness(answer, ctx_strs)
        except Exception: faith = 0.0
        try: relev = self._custom._answer_relevancy(question, answer)
        except Exception: relev = 0.0
        try: cprec = self._custom._context_precision(question, ctx_strs)
        except Exception: cprec = 0.0
        try: crec = self._custom._context_recall(answer, ctx_strs)
        except Exception: crec = 0.0

        faith, relev, cprec, crec = (round(float(x), 3) for x in (faith, relev, cprec, crec))
        overall = round((faith + relev + cprec + crec) / 4.0, 3)
        return RAGMetrics(faith, relev, cprec, crec, overall, backend="custom")
