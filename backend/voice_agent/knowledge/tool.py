"""The ``search_knowledge_base`` function the LLM can call during a conversation."""

from __future__ import annotations

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
- Answer only from what the tool returns, in the caller's language, in one to three short \
spoken sentences. Do not read out URLs character by character; say the site name instead.
- If the tool returns nothing relevant, say you do not have that information and suggest \
calling the helpline, instead of guessing.
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
                        "fees, timelines or complaints."
                    ),
                    properties={
                        "question": {
                            "type": "string",
                            "description": "The caller's question, in their own words.",
                        }
                    },
                    required=["question"],
                )
            ]
        )

    async def handle(self, params: FunctionCallParams) -> None:
        question = str(params.arguments.get("question") or "").strip()
        if not question:
            await params.result_callback({"found": False, "reason": "empty question"})
            return
        logger.info(f"Knowledge: LLM asked to search for {question!r}")
        try:
            result = await self.service.search(question, limit=self.limit)
        except Exception as exc:
            detail = f"{exc} (qdrant={self.service.qdrant_url})"
            logger.warning(f"Knowledge: search failed: {detail}")
            await self._notify(SearchRecord(question, None, 0, 0, 0, [], error=detail[:300]))
            await params.result_callback({"found": False, "reason": "search unavailable"})
            return

        if result.translated_query:
            logger.info(
                f"Knowledge: translated to English in {result.translate_ms} ms -> "
                f"{result.translated_query!r}"
            )
        else:
            logger.info("Knowledge: question already in English, no translation needed")
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

        hits = [h for h in result.hits if h.score >= MIN_SCORE]
        if not hits:
            logger.info(
                f"Knowledge: nothing above the {MIN_SCORE} relevance floor; telling the LLM"
            )
            await params.result_callback({"found": False, "reason": "no relevant information"})
            return
        passages = []
        used_chars = 0
        for hit in hits:
            text = hit.text.strip()
            if used_chars + len(text) > MAX_RESULT_CHARS:
                text = text[: max(0, MAX_RESULT_CHARS - used_chars)]
            if not text:
                break
            passages.append(
                {"source": hit.document_name, "score": round(hit.score, 2), "text": text}
            )
            used_chars += len(text)
        logger.info(
            f"Knowledge: passing {len(passages)} passage(s), {used_chars} chars, to the LLM"
        )
        await params.result_callback(
            {"found": True, "query_in_english": result.translated_query, "passages": passages}
        )
