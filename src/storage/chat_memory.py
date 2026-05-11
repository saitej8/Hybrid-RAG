"""src/storage/chat_memory.py — SQLite chat memory (per-user scoped)."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import settings
from src.core import get_logger

log = get_logger(__name__)


class ChatMemory:
    """
    Per-user chat store. Every chat-scoped operation requires a `user_id`
    so users only ever see their own conversations.

    Message-level operations (set_feedback) are addressable by msg_id but
    still verify ownership through the parent chat.
    """

    def __init__(self, db_path: Union[str, Path, None] = None):
        self.db_path = Path(db_path) if db_path else settings.chat_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        # Order matters: create tables first, then run column migration on
        # any pre-existing 'chats' table, THEN create the index that
        # references user_id. Otherwise older DBs (no user_id column) fail
        # at index creation before the migration runs.
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id    TEXT PRIMARY KEY,
                    user_id    TEXT,
                    title      TEXT NOT NULL,
                    doc_meta   TEXT,
                    created_at REAL,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    msg_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id       TEXT NOT NULL,
                    role          TEXT NOT NULL,
                    content       TEXT NOT NULL,
                    input_tokens  INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    metrics       TEXT,
                    feedback      TEXT,
                    created_at    REAL,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
                );
                CREATE INDEX IF NOT EXISTS idx_msgs_chat ON messages(chat_id, msg_id);
            """)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(chats)").fetchall()}
            if "user_id" not in cols:
                c.execute("ALTER TABLE chats ADD COLUMN user_id TEXT")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_chats_user "
                "ON chats(user_id, updated_at DESC)"
            )

    # ---- chats ----------------------------------------------------------

    def create_chat(self, user_id: str, title: str = "New Chat") -> str:
        if not user_id:
            raise ValueError("user_id is required to create a chat")
        cid = str(uuid.uuid4())
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO chats(chat_id,user_id,title,doc_meta,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (cid, user_id, title, json.dumps({}), now, now),
            )
        log.info("Created chat %s for user %s ('%s')", cid[:8], user_id[:8], title)
        return cid

    def list_chats(self, user_id: str) -> List[Dict[str, Any]]:
        if not user_id:
            return []
        with self._conn() as c:
            rows = c.execute(
                "SELECT chat_id,title,created_at,updated_at FROM chats "
                "WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _owns(self, conn: sqlite3.Connection, chat_id: str, user_id: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM chats WHERE chat_id=? AND user_id=?",
            (chat_id, user_id),
        ).fetchone()
        return row is not None

    def rename_chat(self, chat_id: str, user_id: str, title: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE chats SET title=?, updated_at=? WHERE chat_id=? AND user_id=?",
                (title, time.time(), chat_id, user_id),
            )

    def delete_chat(self, chat_id: str, user_id: str) -> bool:
        with self._conn() as c:
            if not self._owns(c, chat_id, user_id):
                return False
            c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
            c.execute("DELETE FROM chats WHERE chat_id=? AND user_id=?", (chat_id, user_id))
        log.info("Deleted chat %s (user %s)", chat_id[:8], user_id[:8])
        return True

    def update_doc_meta(self, chat_id: str, user_id: str, doc_meta: dict) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE chats SET doc_meta=?, updated_at=? WHERE chat_id=? AND user_id=?",
                (json.dumps(doc_meta), time.time(), chat_id, user_id),
            )

    def get_chat(self, chat_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM chats WHERE chat_id=? AND user_id=?",
                (chat_id, user_id),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["doc_meta"] = json.loads(d["doc_meta"] or "{}")
        except Exception:
            d["doc_meta"] = {}
        return d

    # ---- messages -------------------------------------------------------

    def add_message(self, chat_id: str, user_id: str, role: str, content: str,
                    input_tokens: int = 0, output_tokens: int = 0,
                    metrics: Optional[dict] = None) -> Optional[int]:
        now = time.time()
        with self._conn() as c:
            if not self._owns(c, chat_id, user_id):
                log.warning("add_message: chat %s not owned by user %s", chat_id[:8], user_id[:8])
                return None
            cur = c.execute(
                "INSERT INTO messages(chat_id,role,content,input_tokens,"
                "output_tokens,metrics,created_at) VALUES (?,?,?,?,?,?,?)",
                (chat_id, role, content, input_tokens, output_tokens,
                 json.dumps(metrics) if metrics else None, now),
            )
            c.execute("UPDATE chats SET updated_at=? WHERE chat_id=?", (now, chat_id))
            return cur.lastrowid

    def get_messages(self, chat_id: str, user_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            if not self._owns(c, chat_id, user_id):
                return []
            rows = c.execute(
                "SELECT * FROM messages WHERE chat_id=? ORDER BY msg_id ASC",
                (chat_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else None
            except Exception:
                d["metrics"] = None
            out.append(d)
        return out

    def set_feedback(self, msg_id: int, user_id: str, feedback: Optional[str]) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT m.msg_id FROM messages m "
                "JOIN chats c ON c.chat_id = m.chat_id "
                "WHERE m.msg_id=? AND c.user_id=?",
                (msg_id, user_id),
            ).fetchone()
            if not row:
                return False
            c.execute("UPDATE messages SET feedback=? WHERE msg_id=?", (feedback, msg_id))
        return True

    def short_term_history(self, chat_id: str, user_id: str,
                           last_n: int = 6) -> List[Dict[str, str]]:
        msgs = self.get_messages(chat_id, user_id)
        return [{"role": m["role"], "content": m["content"]} for m in msgs[-last_n:]]

    def export_chat(self, chat_id: str, user_id: str) -> Dict[str, Any]:
        return {
            "chat": self.get_chat(chat_id, user_id) or {},
            "messages": self.get_messages(chat_id, user_id),
        }
