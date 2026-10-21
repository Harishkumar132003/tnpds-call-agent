"""Sarvam pronunciation dictionaries, managed from the portal.

A dictionary maps exact written forms to the text the voice should say instead
(plain text, no phonetic alphabet), per language. Sarvam applies it when a
``dict_id`` is passed to TTS. Limits (per Sarvam): 10 dictionaries per account,
100 entries each, exact matches only.

The app keeps the entries in its own config versions and recreates the Sarvam
dictionary on every save, so removing an entry here also removes it there
(Sarvam's update endpoint only adds or changes entries).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger
from sarvamai import AsyncSarvamAI

MAX_ENTRIES = 100
DICTIONARY_NAME_PREFIX = "voice-agent"


@dataclass(frozen=True)
class PronunciationEntry:
    word: str
    spoken: str


def parse_entries(text: str) -> list[PronunciationEntry]:
    """Parse ``word = spoken form`` lines (``#`` comments and blank lines ignored).

    Raises ``ValueError`` naming the offending line.
    """
    entries: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"line {number}: expected 'word = spoken form', got {line!r}")
        word, spoken = (part.strip() for part in line.split("=", 1))
        if not word or not spoken:
            raise ValueError(f"line {number}: both the word and its spoken form are required")
        entries[word] = spoken
    if len(entries) > MAX_ENTRIES:
        raise ValueError(f"Sarvam allows at most {MAX_ENTRIES} entries per dictionary")
    return [PronunciationEntry(w, s) for w, s in entries.items()]


def entries_to_text(entries: list[dict[str, str]] | None) -> str:
    return "\n".join(f"{e['word']} = {e['spoken']}" for e in (entries or []))


def dictionary_file(language: str, entries: list[PronunciationEntry]) -> bytes:
    """The JSON document Sarvam expects, for one language."""
    return json.dumps(
        {"pronunciations": {language: {e.word: e.spoken for e in entries}}}, ensure_ascii=False
    ).encode()


class SarvamDictionaries:
    def __init__(self, api_key: str) -> None:
        self._client = AsyncSarvamAI(api_subscription_key=api_key)

    async def create(self, language: str, entries: list[PronunciationEntry]) -> str:
        response = await self._client.pronunciation_dictionary.create(
            file=("dictionary.json", dictionary_file(language, entries), "application/json")
        )
        logger.info(
            f"Sarvam dictionary {response.dictionary_id} created "
            f"({len(entries)} entries, {language})"
        )
        return response.dictionary_id

    async def delete(self, dict_id: str) -> None:
        try:
            await self._client.pronunciation_dictionary.delete(dict_id=dict_id)
            logger.info(f"Sarvam dictionary {dict_id} deleted")
        except Exception as exc:  # already gone or not ours: not fatal
            logger.warning(f"Could not delete Sarvam dictionary {dict_id}: {exc}")

    async def entries(self, dict_id: str) -> dict[str, dict[str, str]]:
        data = await self._client.pronunciation_dictionary.get(dict_id)
        return dict(data.pronunciations or {})

    async def replace(
        self, language: str, entries: list[PronunciationEntry], previous_dict_id: str | None
    ) -> str | None:
        """Create a dictionary for ``entries`` and drop the previous one. None if no entries."""
        new_id = await self.create(language, entries) if entries else None
        if previous_dict_id and previous_dict_id != new_id:
            await self.delete(previous_dict_id)
        return new_id
