"""BM25 sparse vectors for Qdrant keyword search, computed in process.

Each chunk is turned into ``{term id: weight}``: lowercase words, English stop words
dropped, simple endings stripped, a stable 32-bit id per term and BM25's term-frequency
saturation as the weight. Qdrant applies the IDF part itself (``Modifier.IDF`` on the
sparse vector), so the dot product it ranks by is a BM25 score. No model download.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

from qdrant_client import models

K1 = 1.2
B = 0.75
AVG_LEN = 256.0  # tokens; fixed, like fastembed's Qdrant/bm25

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORD_TEXT = """
    a an and are as at be been by can could did do does for from
    had has have he her his how i if in into is it its me my
    of on or our she so than that the their them then there these they this
    to was we were what when where which who whom why will with would you your
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())  # noqa: SIM905


def stem(word: str) -> str:
    """Strip a few common English endings so 'cards', 'card' and 'carded' meet."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS]


def term_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF


def encode_document(text: str) -> models.SparseVector:
    """Sparse vector of a stored chunk: BM25 term-frequency saturation per term."""
    tokens = tokenize(text)
    counts = Counter(tokens)
    norm = K1 * (1 - B + B * len(tokens) / AVG_LEN)
    weights = {term_id(t): tf * (K1 + 1) / (tf + norm) for t, tf in counts.items()}
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[i] for i in indices])


def encode_query(text: str) -> models.SparseVector:
    """Sparse vector of a query: one unit per distinct term (IDF is added by Qdrant)."""
    indices = sorted({term_id(t) for t in tokenize(text)})
    return models.SparseVector(indices=indices, values=[1.0] * len(indices))
