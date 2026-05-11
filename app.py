"""
app.py — Entry point.   Run:   streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---- 1. set_page_config MUST be the first Streamlit call ------------
import streamlit as st

st.set_page_config(
    page_title="HybridRAG",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 2. Make `src` importable ---------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- 3. Bootstrap config + logging ----------------------------------
from src.config import settings
from src.core import init_logging, get_logger

settings.ensure_dirs()
init_logging()

log = get_logger("app")
log.info(
    "Starting HybridRAG (model=%s, embedding=%s, streaming=%s, evaluation=%s)",
    settings.gemini_model,
    settings.embedding_model,
    settings.enable_streaming,
    settings.enable_evaluation,
)

# ---- 4. Launch UI ---------------------------------------------------
from src.ui import run_ui
run_ui()
