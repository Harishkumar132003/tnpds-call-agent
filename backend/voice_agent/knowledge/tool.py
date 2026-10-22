"""The ``search_knowledge_base`` function the LLM can call during a conversation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from voice_agent.knowledge.service import KnowledgeService

TOOL_NAME = "search_knowledge_base"
MAX_RESULT_CHARS = 12_000  # a whole FAQ sheet (largest is ~9.5k chars)
MIN_SCORE = 0.25
MAX_QUESTIONS = 4  # searched in parallel per tool call

# Spoken the moment the LLM decides to search, so the caller is never left in silence.
SEARCHING_LINES = {
    "ta-IN": "ஒரு நிமிடம், நான் பார்க்கிறேன்.",
    "hi-IN": "एक मिनट, मैं देखती हूँ।",
    "en-IN": "One moment, let me check.",
    "bn-IN": "এক মিনিট, আমি দেখছি।",
    "gu-IN": "એક મિનિટ, હું જોઉં છું.",
    "kn-IN": "ಒಂದು ನಿಮಿಷ, ನಾನು ನೋಡುತ್ತೇನೆ.",
    "ml-IN": "ഒരു നിമിഷം, ഞാൻ നോക്കട്ടെ.",
    "mr-IN": "एक मिनिट, मी बघते.",
    "od-IN": "ଗୋଟିଏ ମିନିଟ୍, ମୁଁ ଦେଖୁଛି।",
    "pa-IN": "ਇੱਕ ਮਿੰਟ, ਮੈਂ ਦੇਖਦੀ ਹਾਂ।",
    "te-IN": "ఒక్క నిమిషం, నేను చూస్తాను.",
}

PROMPT_ADDITION = """\

Knowledge base:
- You have a tool called search_knowledge_base. Call it for any question about procedures, \
required documents, eligibility, fees, timelines, status checks or complaints, before answering.
- Put each separate topic the caller raised in this turn as one item in "questions", copied \
in the caller's own words and language, never in English unless the caller spoke English. Do \
not translate, shorten or reinterpret. Keep a complaint and its details together as one item; \
split only genuinely different questions (for example steps and documents). Each item must \
make sense on its own: when a question leans on an earlier one ("and what documents?"), add the \
missing subject from the conversation, in the caller's language ("புது ஸ்மார்ட் கார்டுக்கு \
என்னென்ன டாக்குமெண்ட்லாம் தேவை?").
- Answer every question the caller asked, only from what the tool returns, in the caller's \
language: one to three short spoken sentences per question, the key steps only, then offer \
more detail. Do not read out URLs character by character; say the site name instead.
- Speech recognition sometimes mishears words. If a word does not make sense for a ration \
card helpline (for example "ஓட்டோ", "மோடரேஷன் கார்டு"), do not guess what it means and do \
not answer about it: ask the caller to say it again in other words.
- If the tool returns nothing relevant, first ask the caller to say the question again in \
other words. Only if it is still not found, say you do not have that information and suggest \
calling the helpline.
"""


def searching_line(language_code: str) -> str:
    return SEARCHING_LINES.get(language_code, SEARCHING_LINES["en-IN"])


@dataclass(frozen=True)
class SearchRecord:
    """Everything one knowledge search did, for the call log and the portal."""

    question: str
    translated_query: str | None
    translate_ms: int
    embed_ms: int
    search_ms: int
    hits: list[dict[str, Any]]
    """``{"source", "chunk_index", "score", "text", "used"}`` per retrieved chunk."""
    error: str | None = None
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "translated_query": self.translated_query,
            "translate_ms": self.translate_ms,
            "embed_ms": self.embed_ms,
            "search_ms": self.search_ms,
            "hits": self.hits,
            "error": self.error,
            "at": self.at,
        }


SearchListener = Callable[[SearchRecord], Awaitable[None] | None]


@dataclass
class KnowledgeTool:
    """Bridges Pipecat function calls to :class:`KnowledgeService`."""

    service: KnowledgeService
    limit: int = 3
    listeners: list[SearchListener] = field(default_factory=list)

    def on_search(self, listener: SearchListener) -> None:
        """Register a callback that receives every :class:`SearchRecord`."""
        self.listeners.append(listener)

    async def _notify(self, record: SearchRecord) -> None:
        for listener in self.listeners:
            try:
                outcome = listener(record)
                if outcome is not None:
                    await outcome
            except Exception as exc:  # tracking must never break the call
                logger.warning(f"Knowledge search listener failed: {exc}")

    def schema(self) -> ToolsSchema:
        return ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name=TOOL_NAME,
                    description=(
                        "Search the organisation's FAQ and procedure documents. Use for any "
                        "question about how to apply, documents required, status, corrections, "
                        "fees, timelines or complaints. Call it once per caller turn with all "
                        "of the caller's questions."
                    ),
                    properties={
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "One item per separate topic the caller raised, copied in the "
                                "caller's own words and language exactly as they said them "
                                "(Tamil stays Tamil). Do not translate or reword. Keep one "
                                "complaint with its details as a single item. Each item must "
                                "stand on its own: add the subject when a question refers back."
                            ),
                        }
                    },
                    required=["questions"],
                )
            ]
        )

    @staticmethod
    def _questions(arguments: dict[str, Any]) -> list[str]:
        """The questions to search: the ``questions`` list, or a legacy ``question`` string."""
        raw = arguments.get("questions")
        if isinstance(raw, str):
            raw = [raw]
        if not raw and arguments.get("question"):
            raw = [arguments["question"]]
        seen: list[str] = []
        for item in raw or []:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.append(text)
        return seen[:MAX_QUESTIONS]

    async def _search_one(self, question: str) -> tuple[str, Any, str | None]:
        try:
            return question, await self.service.search(question, limit=self.limit), None
        except Exception as exc:
            return question, None, f"{exc} (qdrant={self.service.qdrant_url})"

    async def _record(self, question: str, result: Any, error: str | None) -> list[Any]:
        """Log and notify one question's search; return its hits above the floor."""
        if error is not None:
            logger.warning(f"Knowledge: search failed for {question!r}: {error}")
            await self._notify(SearchRecord(question, None, 0, 0, 0, [], error=error[:300]))
            return []
        if result.translated_query:
            logger.info(
                f"Knowledge: {question!r} translated in {result.translate_ms} ms -> "
                f"{result.translated_query!r}"
            )
        else:
            logger.info(f"Knowledge: {question!r} already in English, no translation needed")
        logger.info(
            f"Knowledge: embedded in {result.embed_ms} ms, Qdrant search in {result.search_ms} ms, "
            f"{len(result.hits)} chunk(s) returned"
        )
        hits_json: list[dict[str, Any]] = []
        for rank, hit in enumerate(result.hits, start=1):
            used = hit.score >= MIN_SCORE
            preview = hit.text.strip().replace("\n", " ")[:110]
            logger.info(
                f"Knowledge:   {rank}. {hit.score:.2f} {'USE ' if used else 'skip'} "
                f"{hit.document_name} #{hit.chunk_index}: {preview!r}"
            )
            hits_json.append(
                {
                    "source": hit.document_name,
                    "document_id": hit.document_id,
                    "chunk_index": hit.chunk_index,
                    "score": round(hit.score, 3),
                    "text": hit.text,
                    "used": used,
                }
            )
        await self._notify(
            SearchRecord(
                question,
                result.translated_query,
                result.translate_ms,
                result.embed_ms,
                result.search_ms,
                hits_json,
            )
        )
        return [h for h in result.hits if h.score >= MIN_SCORE]

    async def handle(self, params: FunctionCallParams) -> None:
        questions = self._questions(params.arguments or {})
        if not questions:
            await params.result_callback({"found": False, "reason": "empty question"})
            return
        logger.info(f"Knowledge: LLM asked to search for {questions!r}")
        # One search per question, in parallel: translation + embedding + Qdrant each
        outcomes = await asyncio.gather(*(self._search_one(q) for q in questions))

        per_question: list[dict[str, Any]] = []
        passages: list[dict[str, Any]] = []
        sent: set[tuple[str, int]] = set()
        used_chars = 0
        failures = 0
        for question, result, error in outcomes:
            failures += error is not None
            hits = await self._record(question, result, error)
            sources = []
            for hit in hits:
                sources.append(hit.document_name)
                key = (hit.document_id, hit.chunk_index)
                if key in sent:  # the same sheet answers two questions: send it once
                    continue
                text = hit.text.strip()
                if used_chars + len(text) > MAX_RESULT_CHARS:
                    text = text[: max(0, MAX_RESULT_CHARS - used_chars)]
                if not text:
                    continue
                sent.add(key)
                passages.append(
                    {"source": hit.document_name, "score": round(hit.score, 2), "text": text}
                )
                used_chars += len(text)
            per_question.append(
                {
                    "question": question,
                    "query_in_english": result.translated_query if result else None,
                    "found": bool(sources),
                    "sources": sources,
                }
            )

        if failures == len(outcomes):
            await params.result_callback({"found": False, "reason": "search unavailable"})
            return
        if not passages:
            logger.info(
                f"Knowledge: nothing above the {MIN_SCORE} relevance floor; telling the LLM"
            )
            await params.result_callback(
                {"found": False, "reason": "no relevant information", "questions": per_question}
            )
            return
        logger.info(
            f"Knowledge: passing {len(passages)} passage(s), {used_chars} chars, "
            f"for {len(questions)} question(s) to the LLM"
        )
        await params.result_callback(
            {"found": True, "questions": per_question, "passages": passages}
        )
