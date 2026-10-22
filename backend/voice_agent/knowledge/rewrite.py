"""Context-aware query rewrite for knowledge search (``KNOWLEDGE_QUERY_REWRITE=true`` only).

Reads the caller's last messages (caller only, oldest first) and writes ONE standalone
English search question with everything the caller wants right now. The prompt is editable on the
portal's Configuration page; :data:`DEFAULT_REWRITE_PROMPT` is used when none is saved.
When the rewrite fails or times out, the tool falls back to translating the questions.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

SARVAM_CHAT_URL = "https://api.sarvam.ai/v2/chat/completions"
HISTORY_MESSAGES = 10

DEFAULT_REWRITE_PROMPT = """\
You write the search question for an English FAQ about Tamil Nadu ration cards (smart cards).
You get the caller's earlier messages (context only) and the caller's LATEST message.
Write ONE standalone English question that contains everything the caller wants right now:
- Cover exactly what is asked in the LATEST message, nothing more. If it asks several \
things, put all of them in this one question. Never add a topic the caller did not ask \
about (for example do not add "documents" or "steps" unless the caller asked for them).
- Use the earlier messages only to fill in what the latest message refers to (after "my \
card is lost", "how much does it cost?" means "fee for lost smart card reprint"). Do not \
include earlier questions the caller is no longer asking about.
- Keep every specific thing the caller said (name, member, mobile number, card type, \
commodity...). Never replace a word the caller said with a different one.
- Use the FAQ's own terms: closing or cancelling a whole card -> "surrender smart card"; \
a card that stopped working -> "inactive card"; taking a person off a card -> "remove \
member"; a new card -> "new smart card application"; a lost card -> "lost smart card \
reprint"; a damaged card -> "damaged smart card reprint"; changing details -> "correction"; \
an elderly or sick person who cannot go to the shop -> "authorization form to collect \
ration for elderly".
- Keep it under 25 words. Do not add place names or the words "Tamil Nadu ration card".
- Output only that one question on one line, nothing else."""


@dataclass(frozen=True)
class Rewrite:
    query: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def history_from_messages(messages: list[Any], limit: int = HISTORY_MESSAGES) -> list[str]:
    """The caller's last ``limit`` text messages from an LLM context, oldest first."""
    said: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):  # multi-part content: keep the text parts
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        if isinstance(content, str) and content.strip():
            said.append(content.strip()[:600])
    return said[-limit:]


def rewrite_input(caller_messages: list[str]) -> str:
    """Earlier messages as context, the latest one on its own."""
    *earlier, latest = caller_messages
    context = "\n".join(f"- {m}" for m in earlier) or "(none)"
    return (
        f"Earlier caller messages (context only, oldest first):\n{context}\n\n"
        f"LATEST caller message:\n{latest}\n\nSearch question:"
    )


def _query(text: str) -> str:
    """The first non-empty line, without numbering, bullets or quotes."""
    for line in text.splitlines():
        query = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if query:
            return query
    return ""


class QueryRewriter:
    def __init__(self, api_key: str, *, model: str = "sarvam-105b", timeout: float = 3.0) -> None:
        self._api_key = api_key
        self.model = model
        self._timeout = timeout

    async def rewrite(
        self, caller_messages: list[str], prompt: str | None = None
    ) -> Rewrite | None:
        """One search question for what the caller wants now, or None when it fails."""
        if not caller_messages:
            return None
        user = rewrite_input(caller_messages)
        body = {
            "model": self.model,
            "reasoning_effort": "none",
            "temperature": 0,
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": (prompt or "").strip() or DEFAULT_REWRITE_PROMPT},
                {"role": "user", "content": user},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await asyncio.wait_for(
                    client.post(
                        SARVAM_CHAT_URL,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=body,
                    ),
                    timeout=self._timeout,
                )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(f"Query rewrite failed, falling back to translation: {exc!r}")
            return None
        query = _query(data["choices"][0]["message"].get("content") or "")
        if not query:
            return None
        usage = data.get("usage") or {}
        return Rewrite(
            query,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )
