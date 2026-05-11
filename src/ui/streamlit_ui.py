"""
src/ui/streamlit_ui.py
======================
Production Streamlit UI for HybridRAG.

Auth gate:
  * If no user is signed in, the login/register screen is rendered.
  * Once signed in, every chat operation is scoped to that user.

Sidebar (signed in):
  * Signed-in user badge + "Sign out"
  * Google API key textbox (overrides .env at runtime)
  * Connected key count + key rotation status
  * "+ New Chat"
  * Conversation list with three-dots popover (Rename / Delete)
  * Real-time streaming toggle
  * Download chat (JSON)

Center:
  * Upload screen: PDF uploader
  * On upload: toast "<filename> uploaded successfully" + token + page stats
  * Three quick-action buttons (Explain / Summarize / Ask Anything)
  * Streaming responses with input/output token counts
  * 👍 Like / 👎 Dislike / 📋 Copy
  * RAG evaluation metrics (ragas or custom)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import build_rag_agent, run_agent, stream_agent
from src.config import settings
from src.core import APIKeyManager, count_tokens, get_logger
from src.evaluation import RAGEvaluator
from src.llm import RotatingChatLLM
from src.retrieval import (
    HybridRetriever, RotatingEmbeddings,
    chunk_documents, delete_vector_store, load_pdf,
)
from src.storage import ChatMemory
from src.ui.auth_ui import (
    current_user_id, current_username, is_logged_in, logout, render_auth_screen,
)

log = get_logger(__name__)

PAGE_CSS = """
<style>
.stApp { background-color: #ffffff; }
section[data-testid="stSidebar"] { background-color: #f8fafc; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }
div[data-testid="stMetricValue"] { font-size: 1.05rem; }
.stChatMessage { border-radius: 12px; }
</style>
"""


@st.cache_resource(show_spinner=False)
def _cached_memory() -> ChatMemory:
    return ChatMemory(settings.chat_db_path)


def _build_pipeline(api_keys: list[str]):
    km = APIKeyManager(api_keys)
    embeddings = RotatingEmbeddings(km)
    llm = RotatingChatLLM(km)
    evaluator = RAGEvaluator(llm=llm, embeddings=embeddings,
                             prefer_ragas=True) if settings.enable_evaluation else None
    return km, embeddings, llm, evaluator


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("memory", _cached_memory())
    ss.setdefault("api_key_input", "")
    ss.setdefault("current_chat_id", None)
    ss.setdefault("retriever", None)
    ss.setdefault("doc_meta", None)
    ss.setdefault("streaming", settings.enable_streaming)


def _resolved_keys() -> list[str]:
    pasted = (st.session_state.get("api_key_input") or "").strip()
    if pasted:
        return [k.strip() for k in pasted.split(",") if k.strip()]
    return settings.all_default_keys()


def _get_pipeline():
    keys = _resolved_keys()
    if not keys:
        return None
    sig = "|".join(keys)
    if st.session_state.get("_pipeline_sig") != sig:
        km, embeddings, llm, evaluator = _build_pipeline(keys)
        agent = build_rag_agent(llm=llm, evaluator=evaluator)
        st.session_state["_pipeline_sig"] = sig
        st.session_state["_km"] = km
        st.session_state["_embeddings"] = embeddings
        st.session_state["_llm"] = llm
        st.session_state["_evaluator"] = evaluator
        st.session_state["_agent"] = agent
    return {
        "km": st.session_state["_km"],
        "embeddings": st.session_state["_embeddings"],
        "llm": st.session_state["_llm"],
        "evaluator": st.session_state["_evaluator"],
        "agent": st.session_state["_agent"],
    }


def _reset_doc_state() -> None:
    for k in ("retriever", "doc_meta"):
        st.session_state[k] = None


def _restore_chat(chat_id: str) -> bool:
    pipe = _get_pipeline()
    if not pipe:
        return False
    uid = current_user_id()
    try:
        retriever = HybridRetriever.load(chat_id, pipe["embeddings"])
        if retriever is None:
            return False
        st.session_state.retriever = retriever
        ch = st.session_state.memory.get_chat(chat_id, uid)
        if ch and ch.get("doc_meta"):
            st.session_state.doc_meta = ch["doc_meta"]
        return True
    except Exception as e:
        log.error("Restore chat %s failed: %s", chat_id[:8], e)
        return False


def _ingest(docs, source_label: str) -> None:
    pipe = _get_pipeline()
    if not pipe:
        st.error("Add a Google API key in the sidebar (or set GOOGLE_API_KEY in .env).")
        return

    with st.status("Chunking document…", expanded=False):
        ck = chunk_documents(docs)

    with st.status(f"Building hybrid index (TF-IDF + FAISS) over {ck.chunk_count} chunks…",
                   expanded=False):
        retriever = HybridRetriever(ck.chunks, pipe["embeddings"])

    total_tokens = sum(count_tokens(d.page_content) for d in docs)

    st.session_state.retriever = retriever
    st.session_state.doc_meta = {
        "source": source_label,
        "n_pages": len(docs),
        "n_chunks": ck.chunk_count,
        "total_tokens": total_tokens,
    }

    cid = st.session_state.current_chat_id
    uid = current_user_id()
    if cid and uid:
        st.session_state.memory.update_doc_meta(cid, uid, st.session_state.doc_meta)
        try:
            retriever.save(cid)
        except Exception as e:
            log.error("Persist failed: %s", e)

    st.toast(f"✅ {source_label} uploaded successfully", icon="✅")


def _render_sidebar() -> None:
    mem: ChatMemory = st.session_state.memory
    uid = current_user_id()

    with st.sidebar:
        # --- Account ----------------------------------------------------
        st.markdown(f"### 👤 {current_username() or 'Account'}")
        if st.button("🚪 Sign out", use_container_width=True):
            logout()
            st.rerun()
        st.divider()

        # --- API key ----------------------------------------------------
        st.markdown("### 🔑 Google API Key")
        st.text_input(
            "Paste key(s) — overrides .env. Use commas for multiple.",
            key="api_key_input", type="password",
            label_visibility="collapsed",
            placeholder="AIza...  (or  AIza...,AIza...)",
        )
        keys = _resolved_keys()
        if keys:
            st.caption(f"✅ {len(keys)} key(s) connected")
        else:
            st.caption("⚠️ No API key configured")

        st.divider()

        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            cid = mem.create_chat(uid, title="New Chat")
            st.session_state.current_chat_id = cid
            _reset_doc_state()
            st.rerun()

        st.markdown("### 💬 Conversations")
        chats = mem.list_chats(uid)
        if not chats:
            st.caption("No chats yet. Click **➕ New Chat** above.")

        for ch in chats:
            cid = ch["chat_id"]
            active = cid == st.session_state.current_chat_id
            cols = st.columns([0.78, 0.22])
            label = ("🟢 " if active else "💭 ") + (ch["title"] or "Untitled")[:26]
            if cols[0].button(label, key=f"open_{cid}", use_container_width=True):
                st.session_state.current_chat_id = cid
                _reset_doc_state()
                _restore_chat(cid)
                st.rerun()

            with cols[1].popover("⋮"):
                new_title = st.text_input("Rename", value=ch["title"], key=f"rn_{cid}")
                if st.button("Save", key=f"sv_{cid}", use_container_width=True):
                    mem.rename_chat(cid, uid, new_title or "Untitled")
                    st.rerun()
                if st.button("🗑️ Delete", key=f"del_{cid}",
                             type="primary", use_container_width=True):
                    if mem.delete_chat(cid, uid):
                        delete_vector_store(cid)
                        if st.session_state.current_chat_id == cid:
                            st.session_state.current_chat_id = None
                            _reset_doc_state()
                    st.rerun()

        st.divider()
        st.markdown("### ⚙️ Settings")
        st.session_state.streaming = st.toggle(
            "🔴 Real-time streaming", value=st.session_state.streaming,
        )

        cid = st.session_state.current_chat_id
        if cid:
            export = mem.export_chat(cid, uid)
            st.download_button(
                "⬇️ Download chat (JSON)",
                data=json.dumps(export, indent=2, default=str),
                file_name=f"chat_{cid[:8]}.json",
                mime="application/json",
                use_container_width=True,
            )


def _render_upload_screen() -> None:
    st.markdown("## 📄 Upload your knowledge base")
    st.write("Upload a PDF document to start asking questions.")

    up = st.file_uploader(
        f"PDF file (max {settings.max_pdf_mb} MB)",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if up is not None and st.session_state.doc_meta is None:
        try:
            docs, _ = load_pdf(up, source=up.name)
            cid = st.session_state.current_chat_id
            uid = current_user_id()
            ch = st.session_state.memory.get_chat(cid, uid) if cid else None
            if ch and ch["title"] in (None, "", "New Chat"):
                st.session_state.memory.rename_chat(cid, uid, up.name[:60])
            _ingest(docs, source_label=up.name)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load PDF: {e}")


def _render_metrics(metrics: dict) -> None:
    if not metrics:
        return
    backend = metrics.get("backend", "custom")
    with st.expander(f"📊 RAG Evaluation Metrics  ·  backend: {backend}",
                     expanded=False):
        c = st.columns(5)
        c[0].metric("Faithfulness",  f"{metrics.get('faithfulness', 0):.2f}")
        c[1].metric("Answer Rel.",   f"{metrics.get('answer_relevancy', 0):.2f}")
        c[2].metric("Ctx Precision", f"{metrics.get('context_precision', 0):.2f}")
        c[3].metric("Ctx Recall",    f"{metrics.get('context_recall', 0):.2f}")
        c[4].metric("Overall",       f"{metrics.get('overall', 0):.2f}")
        st.caption(
            "All scores in [0,1]; higher = better. "
            "Faithfulness: % of answer claims entailed by context · "
            "Answer Relevancy: cosine(question, back-translated questions) · "
            "Context Precision: share of retrieved chunks that are relevant · "
            "Context Recall: share of answer claims covered by context."
        )


def _render_assistant_message(msg: dict) -> None:
    uid = current_user_id()
    with st.chat_message("assistant"):
        st.markdown(msg["content"])
        st.caption(
            f"🟦 input: **{msg.get('input_tokens', 0)}** tok  •  "
            f"🟩 output: **{msg.get('output_tokens', 0)}** tok"
        )
        _render_metrics(msg.get("metrics") or {})

        msg_id = msg.get("msg_id")
        if msg_id is None:
            return

        feedback = msg.get("feedback")  # 'like' | 'dislike' | None

        c1, c2, c3, _ = st.columns([0.08, 0.08, 0.10, 0.74])

        # 👍 Like — color shifts to primary when active
        like_label = "👍 Liked" if feedback == "like" else "👍"
        if c1.button(
            like_label, key=f"like_{msg_id}",
            type=("primary" if feedback == "like" else "secondary"),
            use_container_width=True,
        ):
            new = None if feedback == "like" else "like"
            st.session_state.memory.set_feedback(msg_id, uid, new)
            st.rerun()

        # 👎 Dislike — color shifts to primary when active
        dis_label = "👎 Disliked" if feedback == "dislike" else "👎"
        if c2.button(
            dis_label, key=f"dis_{msg_id}",
            type=("primary" if feedback == "dislike" else "secondary"),
            use_container_width=True,
        ):
            new = None if feedback == "dislike" else "dislike"
            st.session_state.memory.set_feedback(msg_id, uid, new)
            st.rerun()

        # 📋 One-click copy via inline JS (works in all modern browsers)
        # We pair the question (previous user msg) with this answer.
        full_text = _build_copy_payload(msg)
        _render_copy_button(msg_id, full_text)


def _render_copy_button(msg_id: int, text: str) -> None:
    """A real one-click copy button using a tiny HTML+JS snippet."""
    safe = (text or "").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components_html = f"""
    <div style="display:inline-block;">
      <button id="copybtn_{msg_id}"
        style="cursor:pointer; padding:0.25rem 0.6rem; border-radius:0.5rem;
               border:1px solid rgba(49,51,63,0.2); background:#ffffff;
               font-size:0.875rem;">
        📋 Copy
      </button>
    </div>
    <script>
      const btn_{msg_id} = document.getElementById("copybtn_{msg_id}");
      const payload_{msg_id} = `{safe}`;
      btn_{msg_id}.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(payload_{msg_id});
          const old = btn_{msg_id}.innerText;
          btn_{msg_id}.innerText = "✅ Copied";
          setTimeout(() => {{ btn_{msg_id}.innerText = old; }}, 1500);
        }} catch (e) {{
          btn_{msg_id}.innerText = "⚠️ Copy failed";
        }}
      }});
    </script>
    """
    st.components.v1.html(components_html, height=42)


def _build_copy_payload(msg: dict) -> str:
    """Pair the assistant message with the previous user question."""
    cid = st.session_state.current_chat_id
    uid = current_user_id()
    if not cid or not uid:
        return msg.get("content", "")
    all_msgs = st.session_state.memory.get_messages(cid, uid)
    user_q = ""
    for m in all_msgs:
        if m["msg_id"] == msg["msg_id"]:
            break
        if m["role"] == "user":
            user_q = m["content"]
    if user_q:
        return f"Q: {user_q}\n\nA: {msg.get('content', '')}"
    return msg.get("content", "")


def _run_query(question: str, intent: str = "ask") -> None:
    cid = st.session_state.current_chat_id
    uid = current_user_id()
    mem: ChatMemory = st.session_state.memory
    retriever = st.session_state.retriever
    if retriever is None:
        st.error("No document loaded yet.")
        return

    pipe = _get_pipeline()
    if not pipe:
        st.error("No Google API key configured.")
        return

    mem.add_message(cid, uid, "user", question, input_tokens=count_tokens(question))
    history = mem.short_term_history(cid, uid, last_n=6)

    if st.session_state.streaming:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            partial = ""
            final_payload = None

            with st.spinner("Searching knowledge base…"):
                gen = stream_agent(
                    llm=pipe["llm"], evaluator=pipe["evaluator"],
                    question=question, retriever=retriever,
                    history=history, intent=intent,
                )

            for ev in gen:
                t = ev.get("type")
                if t == "retrieved":
                    placeholder.markdown("_Generating…_ ▌")
                elif t == "kb_miss":
                    placeholder.markdown("")
                elif t == "token":
                    partial += ev["delta"]
                    placeholder.markdown(partial + " ▌")
                elif t == "final":
                    final_payload = ev
                    placeholder.markdown(ev["answer"])

            if final_payload is None:
                final_payload = {
                    "answer": partial,
                    "input_tokens": count_tokens(partial),
                    "output_tokens": count_tokens(partial),
                    "metrics": {},
                }

            st.caption(
                f"🟦 input: **{final_payload.get('input_tokens', 0)}** tok  •  "
                f"🟩 output: **{final_payload.get('output_tokens', 0)}** tok"
            )
            _render_metrics(final_payload.get("metrics") or {})

        mem.add_message(
            cid, uid, "assistant", final_payload["answer"],
            input_tokens=int(final_payload.get("input_tokens", 0)),
            output_tokens=int(final_payload.get("output_tokens", 0)),
            metrics=final_payload.get("metrics") or {},
        )
        return

    with st.spinner("Thinking…"):
        final = run_agent(pipe["agent"], question=question,
                          retriever=retriever, history=history, intent=intent)
    mem.add_message(
        cid, uid, "assistant", final.get("answer", ""),
        input_tokens=int(final.get("input_tokens", 0)),
        output_tokens=int(final.get("output_tokens", 0)),
        metrics=final.get("metrics", {}),
    )


def _render_chat() -> None:
    cid = st.session_state.current_chat_id
    uid = current_user_id()
    mem: ChatMemory = st.session_state.memory

    meta = st.session_state.doc_meta or {}
    if meta:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("📄 Pages", meta.get("n_pages", "—"))
        b2.metric("🧩 Chunks", meta.get("n_chunks", "—"))
        b3.metric("🔢 PDF tokens", f"{meta.get('total_tokens', 0):,}")
        b4.metric("🔍 Retrieval", "Hybrid")

    st.markdown("#### Quick actions")
    a1, a2, a3 = st.columns(3)
    if a1.button("📖 Explain PDF", use_container_width=True):
        _run_query("Provide a thorough, structured explanation of this document.",
                   intent="explain")
        st.rerun()
    if a2.button("📝 Summarize PDF", use_container_width=True):
        _run_query("Summarize this document in clear bullet points.", intent="summarize")
        st.rerun()
    if a3.button("❓ Ask Anything", use_container_width=True):
        st.toast("Type your question in the chat box ↓")

    st.markdown("---")

    msgs = mem.get_messages(cid, uid) if cid else []
    for m in msgs:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            _render_assistant_message(m)

    user_text = st.chat_input("Ask anything about your document…")
    if user_text:
        if len(user_text) > settings.max_input_chars:
            st.error(f"Input too long ({len(user_text)} chars). Limit: {settings.max_input_chars}.")
            return
        _run_query(user_text, intent="ask")
        st.rerun()


def run_ui() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    if not is_logged_in():
        render_auth_screen()
        return

    _init_state()
    _render_sidebar()

    st.markdown("# 🔎 HybridRAG")
    st.caption("TF-IDF + Dense Retrieval · LangGraph · Gemini · ragas evaluation")

    if not _resolved_keys():
        st.warning("⬅️ Add your Google API key in the sidebar to begin.")

    if st.session_state.current_chat_id is None:
        st.info("⬅️ Click **➕ New Chat** in the sidebar to begin.")
        return

    if st.session_state.doc_meta is None:
        if _resolved_keys():
            _restore_chat(st.session_state.current_chat_id)
    if st.session_state.doc_meta is None:
        _render_upload_screen()
        return

    _render_chat()
