"""Query translation so Indian-language questions retrieve from English documents."""

from __future__ import annotations

import asyncio
import re
from typing import Protocol

from loguru import logger
from sarvamai import AsyncSarvamAI

TRANSLATE_TIMEOUT_SECS = 2.0
_NON_LATIN = re.compile(r"[^\x00-\x7F]")

# Unicode blocks of the scripts the agent supports, mapped to Sarvam language codes.
# Devanagari is shared by Hindi and Marathi; Hindi is the better default.
_SCRIPT_RANGES = [
    ((0x0B80, 0x0BFF), "ta-IN"),
    ((0x0900, 0x097F), "hi-IN"),
    ((0x0980, 0x09FF), "bn-IN"),
    ((0x0A80, 0x0AFF), "gu-IN"),
    ((0x0C80, 0x0CFF), "kn-IN"),
    ((0x0D00, 0x0D7F), "ml-IN"),
    ((0x0B00, 0x0B7F), "od-IN"),
    ((0x0A00, 0x0A7F), "pa-IN"),
    ((0x0C00, 0x0C7F), "te-IN"),
]


def detect_language_code(text: str, default: str = "hi-IN") -> str:
    """Guess the Sarvam language code from the dominant script in ``text``."""
    counts: dict[str, int] = {}
    for ch in text:
        code = ord(ch)
        for (lo, hi), lang in _SCRIPT_RANGES:
            if lo <= code <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    return max(counts, key=counts.get) if counts else default


def needs_translation(text: str) -> bool:
    """True when a meaningful share of the text is outside ASCII (Indic script)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for ch in letters if _NON_LATIN.match(ch))
    return non_latin / len(letters) > 0.3


class Translator(Protocol):
    async def to_english(self, text: str) -> str | None: ...


class SarvamTranslator:
    """Sarvam translate API; returns None on failure so callers fall back to the original.

    ``source_language`` is the conversation language when known (the agent);
    otherwise the script of the text decides (the portal's test box).
    """

    def __init__(
        self,
        api_key: str,
        *,
        source_language: str | None = None,
        model: str = "sarvam-translate:v1",
    ) -> None:
        self._client = AsyncSarvamAI(api_subscription_key=api_key)
        self._model = model
        self._source_language = source_language

    async def to_english(self, text: str) -> str | None:
        source = self._source_language or detect_language_code(text)
        if source == "en-IN":
            return None
        try:
            response = await asyncio.wait_for(
                self._client.text.translate(
                    input=text,
                    source_language_code=source,
                    target_language_code="en-IN",
                    model=self._model,
                ),
                timeout=TRANSLATE_TIMEOUT_SECS,
            )
        except Exception as exc:
            logger.warning(f"Query translation failed, searching with the original text: {exc}")
            return None
        translated = (response.translated_text or "").strip()
        return translated or None
