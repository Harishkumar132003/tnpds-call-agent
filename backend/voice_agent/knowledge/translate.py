"""Query translation so Indian-language questions retrieve from English documents.

The documents are English and the conversation language is fixed per deployment
(``AGENT_LANGUAGE``), so translation is always <that language> -> English; no
language detection is involved.
"""

from __future__ import annotations

import asyncio
import re
from typing import Protocol

from loguru import logger
from sarvamai import AsyncSarvamAI

TRANSLATE_TIMEOUT_SECS = 2.0
_NON_ASCII = re.compile(r"[^\x00-\x7F]")


def needs_translation(text: str) -> bool:
    """True when the text is not already English (a meaningful share of non-ASCII letters).

    The LLM often passes the search question in English already; skipping those
    saves a translate round trip.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    non_ascii = sum(1 for ch in letters if _NON_ASCII.match(ch))
    return non_ascii / len(letters) > 0.3


class Translator(Protocol):
    async def to_english(self, text: str) -> str | None: ...


class SarvamTranslator:
    """Sarvam translate API, fixed source language -> English; None on failure.

    Callers fall back to the original text, so a translation outage never
    blocks a reply.
    """

    def __init__(
        self, api_key: str, *, source_language: str, model: str = "sarvam-translate:v1"
    ) -> None:
        self._client = AsyncSarvamAI(api_subscription_key=api_key)
        self._model = model
        self._source_language = source_language

    async def to_english(self, text: str) -> str | None:
        if self._source_language == "en-IN":
            return None
        try:
            response = await asyncio.wait_for(
                self._client.text.translate(
                    input=text,
                    source_language_code=self._source_language,
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
