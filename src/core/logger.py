"""src/core/logger.py — Rotating-file + console logger."""
from __future__ import annotations

import logging
import logging.handlers
import sys

from src.config import settings

_INITIALIZED = False
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def init_logging() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    settings.ensure_dirs()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    log_path = settings.log_file_path
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024,
                backupCount=5, encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception as e:
            sys.stderr.write(f"[logger] file handler disabled: {e}\n")

    for noisy in ("httpx", "httpcore", "urllib3", "google",
                  "google.generativeai", "google_genai", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _INITIALIZED = True


def get_logger(name: str | None = None) -> logging.Logger:
    if not _INITIALIZED:
        init_logging()
    return logging.getLogger(name or "hybrid_rag")
