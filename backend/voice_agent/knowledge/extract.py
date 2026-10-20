"""Turn uploaded files into plain text."""

from __future__ import annotations

import csv
import io
from pathlib import PurePosixPath

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


class UnsupportedFileError(ValueError):
    """Raised for file types we cannot extract text from."""


def extract_text(filename: str, data: bytes) -> str:
    """Return the document's text, with paragraphs separated by blank lines."""
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix == ".pdf":
        return _from_pdf(data)
    if suffix == ".docx":
        return _from_docx(data)
    if suffix == ".csv":
        return _from_csv(data)
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise UnsupportedFileError(
        f"Unsupported file type {suffix or '(none)'!r}; use one of {supported}"
    )


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _from_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _from_csv(data: bytes) -> str:
    """One line per row, ``header: value`` pairs, so each row is self-describing."""
    text = data.decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return ""
    header, *body = rows
    lines = []
    for row in body:
        pairs = [
            f"{h.strip()}: {v.strip()}" for h, v in zip(header, row, strict=False) if v.strip()
        ]
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)
