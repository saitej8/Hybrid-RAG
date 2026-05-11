"""
src/agent/rag_agent.py
======================
LangGraph workflow:

    user_input
        │
        ▼
   retrieve_hybrid          (TF-IDF + FAISS)
        │
        ▼
   knowledge_check          (low score → "Not in knowledge base")
        │
        ▼
   build_context
        │
        ▼
    generate                (Gemini)
        │
        ▼
    evaluate                (ragas / custom)
        │
        ▼
       END

Plus `stream_agent` for token-by-token UI streaming.
"""
from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.config import settings
from src.core import count_tokens, get_logger
from src.evaluation import RAGEvaluator, RAGMetrics
from src.retrieval import HybridRetriever, ScoredDoc

log = get_logger(__name__)
KB_MIN_SCORE = 0.05

NOT_IN_KB_ANSWER = (
    "❌ **Not available in the knowledge base.**\n\n"
    "I couldn't find any passages in the uploaded document(s) that are "
    "relevant to your question. Please rephrase, ask about a different "
    "topic from the document, or upload a document that covers this."
)


class RAGState(TypedDict, total=False):
    question: str
    history: List[Dict[str, str]]
    retriever: Any
    scored: List[ScoredDoc]
    retrieved_docs: List[Document]
    in_knowledge_base: bool
    context: str
    answer: str
    input_tokens: int
    output_tokens: int
    metrics: Dict[str, float]
    intent: str


_SYSTEM_BASE = (
    "You are a precise assistant answering STRICTLY from the provided CONTEXT. "
    "If the context does not contain enough information, say so clearly. "
    "Cite specific facts from the context. Do not invent information."
)

_INTENT = {
    "explain": (
        "TASK: Provide a clear, structured explanation of the document. "
        "Cover purpose, main concepts, and key arguments. Use plain language."
    ),
    "summarize": (
        "TASK: Summarize the document concisely. Use 6-10 bullet points "
        "covering the main points and key conclusions."
    ),
    "ask": (
        "TASK: Answer the user's question using the CONTEXT. Be precise. "
        "If the question requires reasoning, show it briefly."
    ),
}


def _build_messages(state: RAGState) -> List:
    intent = state.get("intent", "ask")
    instr = _INTENT.get(intent, _INTENT["ask"])
    history = state.get("history") or []
    history_str = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
    ) if history else ""
    user = (
        f"{instr}\n\n"
        f"CONTEXT:\n\"\"\"\n{state.get('context', '')}\n\"\"\"\n\n"
        f"PREVIOUS CONVERSATION:\n{history_str}\n\n"
        f"QUESTION: {state.get('question', '')}"
    )
    return [SystemMessage(content=_SYSTEM_BASE), HumanMessage(content=user)]


# ---------------- nodes ---------------------------------------------------
def _node_retrieve(state: RAGState) -> RAGState:
    retriever: HybridRetriever = state.get("retriever")
    q = state.get("question", "")
    scored: List[ScoredDoc] = []
    if retriever and q:
        try:
            scored = retriever.search(q, k=settings.retrieval_k)
            log.info(
                "Retrieved %d scored docs (top=%.3f)",
                len(scored), scored[0].combined_score if scored else 0.0,
            )
        except Exception as e:
            log.error("Retrieval failed: %s", e)
    docs = [s.document for s in scored]
    return {**state, "scored": scored, "retrieved_docs": docs}


def _node_knowledge_check(state: RAGState) -> RAGState:
    scored: List[ScoredDoc] = state.get("scored") or []
    if not scored:
        log.info("KB guard: no chunks retrieved")
        return {**state, "in_knowledge_base": False}

    top = scored[0]
    # Pass if EITHER the combined score is reasonable OR the raw dense
    # similarity is at least mediocre. After min-max normalization the
    # top combined score can be small even when retrieval is good.
    in_kb = (top.combined_score >= KB_MIN_SCORE) or (top.dense_score >= 0.35)

    log.info(
        "KB guard: top combined=%.3f, dense=%.3f, sparse=%.3f -> %s",
        top.combined_score, top.dense_score, top.sparse_score,
        "PASS" if in_kb else "FAIL",
    )
    return {**state, "in_knowledge_base": in_kb}


def _node_build_context(state: RAGState) -> RAGState:
    if not state.get("in_knowledge_base", False):
        return {**state, "context": ""}
    docs = state.get("retrieved_docs") or []
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "doc")
        page = d.metadata.get("page", "")
        header = f"[{i}] {src}" + (f" (p.{page})" if page else "")
        parts.append(f"{header}\n{d.page_content}")
    return {**state, "context": "\n\n---\n\n".join(parts)}


def _make_generate_node(llm):
    def _node(state: RAGState) -> RAGState:
        if not state.get("in_knowledge_base", False):
            return {**state, "answer": NOT_IN_KB_ANSWER,
                    "input_tokens": count_tokens(state.get("question", "")),
                    "output_tokens": count_tokens(NOT_IN_KB_ANSWER)}
        messages = _build_messages(state)
        in_tok = sum(count_tokens(getattr(m, "content", "")) for m in messages)
        try:
            resp = llm.invoke(messages)
            answer = getattr(resp, "content", str(resp))
        except Exception as e:
            log.exception("LLM generation failed")
            answer = f"⚠️ Generation failed: {e}"
        return {**state, "answer": answer,
                "input_tokens": in_tok, "output_tokens": count_tokens(answer)}
    return _node


def _make_evaluate_node(evaluator: Optional[RAGEvaluator]):
    def _node(state: RAGState) -> RAGState:
        if not state.get("in_knowledge_base", False) or evaluator is None:
            return {**state, "metrics": RAGMetrics.empty().to_dict()}
        try:
            m = evaluator.evaluate(
                question=state.get("question", ""),
                answer=state.get("answer", ""),
                contexts=state.get("retrieved_docs", []) or [],
            )
            log.info("Evaluation overall=%.3f via %s", m.overall, m.backend)
            return {**state, "metrics": m.to_dict()}
        except Exception as e:
            log.error("Eval failed: %s", e)
            return {**state, "metrics": RAGMetrics.empty().to_dict()}
    return _node


# ---------------- graph builder ------------------------------------------
def build_rag_agent(llm, evaluator: Optional[RAGEvaluator] = None):
    g = StateGraph(RAGState)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("knowledge_check", _node_knowledge_check)
    g.add_node("build_context", _node_build_context)
    g.add_node("generate", _make_generate_node(llm))
    g.add_node("evaluate", _make_evaluate_node(evaluator))

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "knowledge_check")
    g.add_edge("knowledge_check", "build_context")
    g.add_edge("build_context", "generate")
    g.add_edge("generate", "evaluate")
    g.add_edge("evaluate", END)
    return g.compile()


# ---------------- sync runner --------------------------------------------
def run_agent(graph, question: str, retriever, history=None, intent="ask") -> Dict[str, Any]:
    return graph.invoke({
        "question": question,
        "history": history or [],
        "retriever": retriever,
        "intent": intent,
    })


# ---------------- streaming runner ---------------------------------------
def stream_agent(
    llm,
    evaluator: Optional[RAGEvaluator],
    question: str,
    retriever: HybridRetriever,
    history: Optional[List[Dict[str, str]]] = None,
    intent: str = "ask",
) -> Generator[Dict[str, Any], None, None]:
    """Yields {type, ...}: 'retrieved' | 'kb_miss' | 'token' | 'final'."""
    history = history or []
    state: RAGState = {
        "question": question, "history": history,
        "retriever": retriever, "intent": intent,
    }
    state = _node_retrieve(state)
    yield {"type": "retrieved", "scored": state.get("scored", [])}

    state = _node_knowledge_check(state)
    if not state.get("in_knowledge_base", False):
        yield {"type": "kb_miss"}
        in_tok = count_tokens(question)
        out_tok = count_tokens(NOT_IN_KB_ANSWER)
        yield {
            "type": "final",
            "answer": NOT_IN_KB_ANSWER,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "metrics": RAGMetrics.empty().to_dict(),
            "scored": state.get("scored", []),
        }
        return

    state = _node_build_context(state)
    messages = _build_messages(state)
    in_tok = sum(count_tokens(getattr(m, "content", "")) for m in messages)

    parts: List[str] = []
    try:
        for chunk in llm.stream(messages):
            delta = getattr(chunk, "content", None) or ""
            if delta:
                parts.append(delta)
                yield {"type": "token", "delta": delta}
    except Exception as e:
        log.exception("Stream LLM failed")
        err = f"\n\n⚠️ Streaming failed: {e}"
        parts.append(err)
        yield {"type": "token", "delta": err}

    answer = "".join(parts)
    out_tok = count_tokens(answer)

    metrics = RAGMetrics.empty().to_dict()
    if evaluator is not None:
        try:
            m = evaluator.evaluate(
                question=question, answer=answer,
                contexts=state.get("retrieved_docs", []) or [],
            )
            metrics = m.to_dict()
            log.info("Stream eval overall=%.3f via %s", m.overall, m.backend)
        except Exception as e:
            log.error("Stream eval failed: %s", e)

    yield {
        "type": "final",
        "answer": answer,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "metrics": metrics,
        "scored": state.get("scored", []),
    }
