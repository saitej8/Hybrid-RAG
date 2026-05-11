"""
src/ui/auth_ui.py
=================
Login + register screens. Renders only when there is no logged-in user
in `st.session_state`. On success it stores `auth_user_id` and
`auth_username` in session state and triggers a rerun so the main UI
takes over.
"""
from __future__ import annotations

import streamlit as st

from src.auth import (
    AuthError,
    AuthManager,
    InvalidCredentials,
    InvalidUsernameError,
    UserExistsError,
    WeakPasswordError,
)
from src.config import settings
from src.core import get_logger

log = get_logger(__name__)


DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo123"


@st.cache_resource(show_spinner=False)
def _cached_auth() -> AuthManager:
    am = AuthManager(settings.chat_db_path)
    am.ensure_demo_account(DEMO_USERNAME, DEMO_PASSWORD)
    return am


def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user_id"))


def current_user_id() -> str | None:
    return st.session_state.get("auth_user_id")


def current_username() -> str | None:
    return st.session_state.get("auth_username")


def logout() -> None:
    for k in (
        "auth_user_id", "auth_username",
        "current_chat_id", "retriever", "doc_meta",
        "_pipeline_sig", "_km", "_embeddings", "_llm", "_evaluator", "_agent",
    ):
        st.session_state.pop(k, None)


def _set_session(user_id: str, username: str) -> None:
    st.session_state["auth_user_id"] = user_id
    st.session_state["auth_username"] = username
    st.session_state["current_chat_id"] = None
    st.session_state["retriever"] = None
    st.session_state["doc_meta"] = None


def render_auth_screen() -> None:
    """Top-level login/register page. Returns after rendering."""
    auth = _cached_auth()

    st.markdown("# 🔎 HybridRAG")
    st.caption("Sign in to start chatting with your documents.")

    tab_login, tab_register = st.tabs(["🔐 Sign in", "🆕 Create account"])

    # --- Sign in --------------------------------------------------------
    with tab_login:
        st.info(
            f"💡 **Demo account:** username `{DEMO_USERNAME}` · "
            f"password `{DEMO_PASSWORD}` — sign in with these to try the app "
            "without registering."
        )
        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("Username", key="login_username",
                              autocomplete="username")
            p = st.text_input("Password", type="password", key="login_password",
                              autocomplete="current-password")
            submit = st.form_submit_button("Sign in", use_container_width=True,
                                           type="primary")
        if submit:
            try:
                uid = auth.login(u, p)
                _set_session(uid, u.strip())
                st.success("Welcome back!")
                st.rerun()
            except InvalidCredentials as e:
                st.error(str(e))
            except AuthError as e:
                st.error(str(e))
            except Exception as e:
                log.exception("Login crashed")
                st.error(f"Login failed: {e}")

    # --- Register -------------------------------------------------------
    with tab_register:
        with st.form("register_form", clear_on_submit=False):
            ru = st.text_input("Choose a username", key="reg_username",
                               help="3–32 chars: letters, digits, '_', '.', '-'")
            rp = st.text_input("Choose a password", type="password",
                               key="reg_password",
                               help="At least 6 characters")
            rp2 = st.text_input("Confirm password", type="password",
                                key="reg_password_confirm")
            submit_r = st.form_submit_button("Create account",
                                             use_container_width=True,
                                             type="primary")
        if submit_r:
            if rp != rp2:
                st.error("Passwords do not match.")
            else:
                try:
                    uid = auth.register(ru, rp)
                    _set_session(uid, ru.strip())
                    st.success("Account created. You're signed in!")
                    st.rerun()
                except (UserExistsError, WeakPasswordError,
                        InvalidUsernameError) as e:
                    st.error(str(e))
                except AuthError as e:
                    st.error(str(e))
                except Exception as e:
                    log.exception("Register crashed")
                    st.error(f"Could not create account: {e}")

    st.divider()
    st.caption(
        "⚠️ Heads-up: this app runs on Streamlit Community Cloud's free tier. "
        "Accounts and chats live on the container's local disk and reset when "
        "the app sleeps or redeploys. Don't reuse a password from anywhere else."
    )
