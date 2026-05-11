"""src/retrieval/pdf_loader.py — PDF ingestion (pdfplumber + pypdf fallback)."""
from __future__ import annotations

import io
from typing import List, Tuple, Union

from langchain_core.documents import Document

from src.config import settings
from src.core import get_logger

log = get_logger(__name__)


def _load_with_pdfplumber(file_bytes: bytes, source: str) -> List[Document]:
    import pdfplumber
    docs: List[Document] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": source, "page": i + 1, "type": "pdf"},
                ))
    return docs


def _load_with_pypdf(file_bytes: bytes, source: str) -> List[Document]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": source, "page": i + 1, "type": "pdf"},
            ))
    return docs


def load_pdf(
    file: Union[bytes, "io.BytesIO"],
    source: str = "uploaded.pdf",
) -> Tuple[List[Document], int]:
    """Returns (documents per page, total page count)."""
    if hasattr(file, "read"):
        file.seek(0)
        file_bytes = file.read()
    else:
        file_bytes = file

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_pdf_mb:
        raise ValueError(
            f"PDF is {size_mb:.1f} MB, exceeds limit of {settings.max_pdf_mb} MB."
        )
    log.info("Loading PDF '%s' (%.2f MB)", source, size_mb)

    docs: List[Document] = []
    try:
        docs = _load_with_pdfplumber(file_bytes, source)
    except Exception as e:
        log.warning("pdfplumber failed (%s) – falling back to pypdf", e)

    if not docs:
        docs = _load_with_pypdf(file_bytes, source)

    if not docs:
        raise ValueError(
            "Could not extract text — PDF may be a scanned image (no OCR in this build)."
        )

    try:
        from pypdf import PdfReader
        n_pages = len(PdfReader(io.BytesIO(file_bytes)).pages)
    except Exception:
        n_pages = max((d.metadata.get("page", 0) for d in docs), default=len(docs))

    log.info("PDF loaded: %d pages, %d with text", n_pages, len(docs))
    return docs, n_pages
