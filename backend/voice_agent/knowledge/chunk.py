"""Split text into overlapping chunks that respect paragraph boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Pack whole paragraphs up to ``max_chars``; oversized ones are split by sentence.

    ``overlap`` characters from the end of a chunk are repeated at the start of
    the next so a fact that straddles a boundary is still retrievable.
    """
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("need max_chars > overlap >= 0")
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
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
