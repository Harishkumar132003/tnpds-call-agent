"""Split text into overlapping chunks that respect paragraph boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")

# A whole document is kept as one chunk up to this size (~6k tokens, well inside
# the 8k-token embedding limit); longer ones fall back to size-based splitting.
WHOLE_DOCUMENT_MAX_CHARS = 24_000


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(
    text: str, *, max_chars: int = 1200, overlap: int = 200, whole_document: bool = False
) -> list[Chunk]:
    """Split ``text`` for embedding.

    ``whole_document=True`` keeps the entire text as a single chunk (the right
    unit for one-topic documents such as an FAQ sheet) as long as it fits
    :data:`WHOLE_DOCUMENT_MAX_CHARS`; otherwise, and by default, whole
    paragraphs are packed up to ``max_chars`` with ``overlap`` characters
    repeated across boundaries, and oversized paragraphs are split by sentence.
    """
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("need max_chars > overlap >= 0")
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
    if whole_document:
        whole = "\n\n".join(paragraphs)
        if not whole:
            return []
        if len(whole) <= WHOLE_DOCUMENT_MAX_CHARS:
            return [Chunk(0, whole)]
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_long(paragraph, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        tail = current[-overlap:].strip() if overlap and current else ""
        current = (
            f"{tail}\n\n{piece}" if tail and len(tail) + len(piece) + 2 <= max_chars else piece
        )
    if current:
        chunks.append(current)
    return [Chunk(i, c) for i, c in enumerate(chunks)]


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        if len(sentence) > max_chars:  # no sentence breaks: hard cut
            if current:
                out.append(current)
                current = ""
            out.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            out.append(current)
            current = sentence
    if current:
        out.append(current)
    return out
