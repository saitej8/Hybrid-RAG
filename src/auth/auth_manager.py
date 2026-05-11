"""
src/auth/auth_manager.py
========================
Bcrypt-backed user accounts stored in the same SQLite database used for
chat memory. Provides:

    AuthManager(db_path).register(username, password) -> user_id
    AuthManager(db_path).login(username, password)    -> user_id
    AuthManager(db_path).get_user(user_id)            -> dict | None

Note on Streamlit Community Cloud: the SQLite file lives on the container's
ephemeral filesystem, so accounts are wiped on restart/redeploy. The schema
and API are deliberately small so they can be ported to Postgres later
without touching call sites.
"""
from __future__ import annotations

import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

import bcrypt

from src.config import settings
from src.core import get_logger

log = get_logger(__name__)

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
_MIN_PASSWORD_LEN = 6


class AuthError(Exception):
    """Base auth error."""


class UserExistsError(AuthError):
    """Username already taken."""


class InvalidCredentials(AuthError):
    """Username/password did not match."""


class WeakPasswordError(AuthError):
    """Password fails minimum requirements."""


class InvalidUsernameError(AuthError):
    """Username fails format rules."""


class AuthManager:
    """
    SQLite-backed user store. Safe to instantiate per request — the
    underlying connection is opened/closed for each operation.
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
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id       TEXT PRIMARY KEY,
                    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at    REAL NOT NULL,
                    last_login_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                """
            )

    @staticmethod
    def _validate_username(username: str) -> str:
        u = (username or "").strip()
        if not _USERNAME_RE.match(u):
            raise InvalidUsernameError(
                "Username must be 3–32 characters: letters, digits, '_', '.', '-'."
            )
        return u

    @staticmethod
    def _validate_password(password: str) -> None:
        if not password or len(password) < _MIN_PASSWORD_LEN:
            raise WeakPasswordError(
                f"Password must be at least {_MIN_PASSWORD_LEN} characters."
            )

    @staticmethod
    def _hash(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _verify(password: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False

    def register(self, username: str, password: str) -> str:
        u = self._validate_username(username)
        self._validate_password(password)
        uid = str(uuid.uuid4())
        now = time.time()
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO users(user_id, username, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (uid, u, self._hash(password), now),
                )
        except sqlite3.IntegrityError as exc:
            raise UserExistsError(f"Username '{u}' is already taken.") from exc
        log.info("Registered user %s (id=%s)", u, uid[:8])
        return uid

    def login(self, username: str, password: str) -> str:
        u = (username or "").strip()
        if not u or not password:
            raise InvalidCredentials("Username and password are required.")
        with self._conn() as c:
            row = c.execute(
                "SELECT user_id, password_hash FROM users WHERE username=? COLLATE NOCASE",
                (u,),
            ).fetchone()
            if not row or not self._verify(password, row["password_hash"]):
                raise InvalidCredentials("Invalid username or password.")
            c.execute(
                "UPDATE users SET last_login_at=? WHERE user_id=?",
                (time.time(), row["user_id"]),
            )
        log.info("Login success for user %s", u)
        return row["user_id"]

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT user_id, username, created_at, last_login_at "
                "FROM users WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def user_exists(self, username: str) -> bool:
        u = (username or "").strip()
        if not u:
            return False
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM users WHERE username=? COLLATE NOCASE",
                (u,),
            ).fetchone()
        return row is not None

    def ensure_demo_account(self, username: str = "demo",
                            password: str = "demo123") -> Optional[str]:
        """Create a default demo account on first run. No-op if it exists."""
        if self.user_exists(username):
            return None
        try:
            uid = self.register(username, password)
            log.info("Bootstrapped demo account '%s'", username)
            return uid
        except AuthError as e:
            log.warning("Could not create demo account: %s", e)
            return None
