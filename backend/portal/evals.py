"""Evaluation: a bank of test questions, run one by one as brand-new chats.

Each question is asked in Tamil in a fresh :class:`chat_agent.session.ChatSession`,
which runs the same steps a call runs (decide, knowledge search, answer). What
came back is stored per question under a batch id: the rewritten English query,
every retrieved chunk with its score and whether it was sent to the model, the
rank of the expected chunk, whether that chunk was sent, and the generated
answer. A batch runs in the background on the portal's event loop, one question
at a time, so the knowledge service and the Sarvam API see the same load as one
caller.

Mounted by the portal under ``/api/evals``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import asyncpg
import httpx
from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from chat_agent.session import ChatSession, ChatTurn
from voice_agent import models as catalogue
from voice_agent.config import Settings
from voice_agent.knowledge.service import KnowledgeService, SearchType
from voice_agent.tracking.config_store import AgentConfig, load_active_config

SARVAM_CHAT_URL = "https://api.sarvam.ai/v2/chat/completions"
TRANSLATE_TIMEOUT_SECS = 20.0
SHEET = re.compile(r"^pb\d+(?:-\d+)?$")

TAMIL_PROMPT = """\
You turn an English helpdesk test question into what a REAL caller would say on the phone to \
the Tamil Nadu ration card helpline. The caller is an ordinary ration card holder, often older \
and not used to websites or official words. Write ONE short spoken Tamil utterance (one or two \
sentences), the way people in Tamil Nadu actually talk on the phone.

Rules:
- Plain everyday spoken Tamil (Chennai / Tamil Nadu colloquial), Tamil script. English words \
only when ordinary people say them: ரேஷன், கார்டு, ஆன்லைன், வெப்சைட், ஆப், OTP, SMS, ஆதார், \
போன், நம்பர், அப்ளை, கம்ப்ளைன்ட், ஆபீஸ், கடை. Write them in Tamil script (OTP and SMS may stay \
in Latin letters).
- NEVER use helpdesk or office words. Callers do not say: reprint, duplicate, mutation, \
resubmit, reopen, entitlement, surrender, portability, authorised person, authorization form, \
card registration, POS, TSO, AC, DSO, IMPDS, EKYC, PMGKAY, SSR, PHH, NPHH, AAY, status, request, \
reference number, application, scheme, procedure, document verification. Describe the situation \
instead: "புது கார்டு வேணும்", "கார்டு கிழிஞ்சு போச்சு", "கார்டை க்ளோஸ் பண்ணணும்", "எவ்வளவு அரிசி \
வரும்", "என் அப்ளிகேஷன் என்ன ஆச்சு", "ஒரு பேப்பர் குடுத்தேன்", "தாலுகா ஆபீஸ்".
- Say "ரேஷன் கார்டு" or just "கார்டு"; say "ஸ்மார்ட் கார்டு" only when the English is about the \
new card itself. Say "ரேஷன் கடை" / "கடைக்காரர்" for the fair price shop / shopkeeper.
- Often start with the situation, then the question, like a real call: "என் கார்டு \
தொலைஞ்சு போச்சு, இப்ப என்ன பண்றது?". Keep the specific facts of the English question \
(who, what happened, what they want). Do not add facts that are not there (do not say the \
application was rejected, or that a name was changed, unless the English says so).
- When the English names an office procedure that has no everyday word, describe what the \
caller wants from it; never swap it for a different request (an authorised person is someone \
collecting ration on the card holder's behalf, not a new member on the card).
- Output only the Tamil utterance, nothing else.

Examples:
English: How do I apply for a reprint of my damaged smart card?
Tamil: என் ரேஷன் கார்டு கிழிஞ்சு போச்சு, புது கார்டு எப்படி வாங்குறது?
English: Why was my add member request reopened?
Tamil: என் பொண்ணு பேரை கார்டுல சேர்க்க ஆன்லைன்ல குடுத்தேன், அது திரும்ப வந்துடுச்சுன்னு மெசேஜ் \
வந்திருக்கு, ஏன்?
English: How do I surrender my smart card?
Tamil: எனக்கு இந்த ரேஷன் கார்டு வேண்டாம், அதை எப்படி க்ளோஸ் பண்றது?
English: What is my entitlement for a PHH rice card with 4 members?
Tamil: எங்க வீட்டுல நாலு பேரு, அரிசி கார்டு, மாசம் எவ்வளவு அரிசி கிடைக்கும்?
English: How do I check the status of my address change request?
Tamil: அட்ரஸ் மாத்த அப்ளை பண்ணேன், அது ஆச்சா இல்லையான்னு எப்படி பாக்குறது?
English: How do I add an authorised person to a smart card online?
Tamil: என் அம்மா வயசானவங்க, கடைக்கு போக முடியல. அவங்க சார்பா நான் ரேஷன் வாங்கறதுக்கு ஆன்லைன்ல \
எப்படி ஏற்பாடு பண்றது?
English: How do I resubmit my smart card application?
Tamil: நான் புது கார்டுக்கு அப்ளை பண்ணேன், திருத்தம் பண்ணி திரும்ப அனுப்பணும்னு மெசேஜ் வந்துச்சு. \
அதை எப்படி பண்றது?"""


# ---------------------------------------------------------------------------
# outcome of one question


@dataclass(frozen=True)
class Outcome:
    """What one run of a question produced, derived from the chat turn."""

    rewritten_query: str | None
    hits: list[dict]
    expected_rank: int | None
    expected_sent: bool
    answer: str
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    ms: int


def source_matches(source: str | None, expected: str) -> bool:
    """A hit's ``source`` is the document name, ``"<sheet> <title>"``; match on the sheet."""
    if not source:
        return False
    return source == expected or source.startswith(expected + " ")


def outcome_of(turn: ChatTurn, expected_source: str) -> Outcome:
    """Rank and sent flag of the expected chunk among everything the turn retrieved."""
    hits: list[dict] = []
    rewritten: str | None = None
    for search in turn.searches:
        rewritten = search.get("rewritten_query") or search.get("translated_query") or rewritten
        for hit in search.get("hits") or []:
            hits.append(
                {
                    "source": hit.get("source"),
                    "document_id": hit.get("document_id"),
                    "chunk_index": hit.get("chunk_index"),
                    "score": hit.get("score"),
                    "used": bool(hit.get("used")),
                }
            )
    rank: int | None = None
    sent = False
    for i, hit in enumerate(hits, start=1):
        if source_matches(hit["source"], expected_source):
            rank, sent = i, hit["used"]
            break
    return Outcome(
        rewritten_query=rewritten,
        hits=hits,
        expected_rank=rank,
        expected_sent=sent,
        answer=turn.reply,
        error=turn.error,
        prompt_tokens=turn.prompt_tokens,
        completion_tokens=turn.completion_tokens,
        ms=turn.ms,
    )


# ---------------------------------------------------------------------------
# storage


class EvalStore(Protocol):
    """The rows the evaluation reads and writes (Postgres in the portal, a fake in tests)."""

    async def questions(
        self, *, offset: int = 0, limit: int = 50, q: str | None = None
    ) -> tuple[list[dict], int]: ...
    async def runnable_questions(self) -> list[dict]: ...
    async def question(self, question_id: int) -> dict | None: ...
    async def add_question(
        self,
        question_en: str,
        question_ta: str | None,
        expected_source: str,
        note: str | None,
        origin: str,
    ) -> dict: ...
    async def update_question(self, question_id: int, **fields: Any) -> dict | None: ...
    async def delete_question(self, question_id: int) -> bool: ...
    async def existing_keys(self) -> set[tuple[str, str]]: ...
    async def untranslated(self) -> list[dict]: ...
    async def clear_tamil(self, origin: str | None) -> int: ...
    async def create_batch(
        self,
        batch_id: str,
        total: int,
        config_id: int | None,
        collection: str | None,
        note: str | None,
        concurrency: int = 1,
        force_search: bool = False,
        search_type: str = "semantic",
    ) -> dict: ...
    async def add_result(self, batch_id: str, question: dict, outcome: Outcome) -> None: ...
    async def finish_batch(self, batch_id: str, status: str, error: str | None) -> None: ...
    async def batches(self) -> list[dict]: ...
    async def batch(self, batch_id: str) -> dict | None: ...
    async def results(self, batch_id: str) -> list[dict]: ...
    async def delete_batch(self, batch_id: str) -> bool: ...


QUESTION_COLUMNS = (
    "id, question_en, question_ta, expected_source, note, enabled, origin, created_at, updated_at"
)
BATCH_COLUMNS = (
    "b.id, b.started_at, b.finished_at, b.status, b.total, b.done, b.config_id, b.collection, "
    "b.note, b.error, b.concurrency, b.force_search, b.search_type"
)
SUMMARY_SQL = """
    count(r.id)                                        as results,
    count(r.expected_rank)                             as retrieved,
    count(r.id) filter (where r.expected_sent)         as sent,
    count(r.id) filter (where r.expected_rank = 1)     as rank1,
    count(r.error)                                     as errors,
    coalesce(round(avg(r.ms))::int, 0)                 as avg_ms
"""


def _json(record: asyncpg.Record | None) -> dict | None:
    if record is None:
        return None
    row = dict(record)
    for key, value in row.items():
        if isinstance(value, datetime):
            row[key] = value.isoformat()
        elif isinstance(value, uuid.UUID):
            row[key] = str(value)
    if isinstance(row.get("hits"), str):
        row["hits"] = json.loads(row["hits"])
    return row


class PostgresEvalStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def questions(
        self, *, offset: int = 0, limit: int = 50, q: str | None = None
    ) -> tuple[list[dict], int]:
        """One page of questions (newest id last) and the total matching ``q``."""
        where, args = "", []
        if q:
            where = (
                "where question_en ilike $1 or question_ta ilike $1 "
                "or expected_source ilike $1 or coalesce(note, '') ilike $1"
            )
            args = [f"%{q}%"]
        async with self._pool.acquire() as conn:
            total = await conn.fetchval(f"select count(*) from eval_questions {where}", *args)
            rows = await conn.fetch(
                f"select {QUESTION_COLUMNS} from eval_questions {where} order by id "
                f"offset ${len(args) + 1} limit ${len(args) + 2}",
                *args,
                offset,
                limit,
            )
        return [_json(r) for r in rows], int(total)

    async def runnable_questions(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"select {QUESTION_COLUMNS} from eval_questions "
                "where enabled and question_ta is not null order by id"
            )
        return [_json(r) for r in rows]

    async def question(self, question_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"select {QUESTION_COLUMNS} from eval_questions where id = $1", question_id
            )
        return _json(row)

    async def add_question(
        self,
        question_en: str,
        question_ta: str | None,
        expected_source: str,
        note: str | None,
        origin: str,
    ) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "insert into eval_questions (question_en, question_ta, expected_source, note, "
                f"origin) values ($1, $2, $3, $4, $5) returning {QUESTION_COLUMNS}",
                question_en,
                question_ta,
                expected_source,
                note,
                origin,
            )
        return _json(row)

    async def update_question(self, question_id: int, **fields: Any) -> dict | None:
        allowed = {"question_en", "question_ta", "expected_source", "note", "enabled"}
        changes = {k: v for k, v in fields.items() if k in allowed}
        if not changes:
            return await self.question(question_id)
        sets = ", ".join(f"{k} = ${i}" for i, k in enumerate(changes, start=2))
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"update eval_questions set {sets}, updated_at = now() where id = $1 "
                f"returning {QUESTION_COLUMNS}",
                question_id,
                *changes.values(),
            )
        return _json(row)

    async def delete_question(self, question_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute("delete from eval_questions where id = $1", question_id)
        return result.endswith(" 1")

    async def existing_keys(self) -> set[tuple[str, str]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("select question_en, expected_source from eval_questions")
        return {(r["question_en"], r["expected_source"]) for r in rows}

    async def untranslated(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"select {QUESTION_COLUMNS} from eval_questions "
                "where question_ta is null order by id"
            )
        return [_json(r) for r in rows]

    async def clear_tamil(self, origin: str | None) -> int:
        """Forget the Tamil text of every question (or of one origin) so it is written again."""
        async with self._pool.acquire() as conn:
            if origin:
                result = await conn.execute(
                    "update eval_questions set question_ta = null, updated_at = now() "
                    "where origin = $1",
                    origin,
                )
            else:
                result = await conn.execute(
                    "update eval_questions set question_ta = null, updated_at = now()"
                )
        return int(result.rsplit(" ", 1)[-1])

    async def create_batch(
        self,
        batch_id: str,
        total: int,
        config_id: int | None,
        collection: str | None,
        note: str | None,
        concurrency: int = 1,
        force_search: bool = False,
        search_type: str = "semantic",
    ) -> dict:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "insert into eval_batches (id, status, total, config_id, collection, note, "
                "concurrency, force_search, search_type) "
                "values ($1, 'running', $2, $3, $4, $5, $6, $7, $8)",
                uuid.UUID(batch_id),
                total,
                config_id,
                collection,
                note,
                concurrency,
                force_search,
                search_type,
            )
        return await self.batch(batch_id)  # type: ignore[return-value]

    async def add_result(self, batch_id: str, question: dict, outcome: Outcome) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "insert into eval_results (batch_id, question_id, question_ta, expected_source, "
                "rewritten_query, hits, expected_rank, expected_sent, answer, error, "
                "prompt_tokens, completion_tokens, ms) "
                "values ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12, $13)",
                uuid.UUID(batch_id),
                question["id"],
                question["question_ta"],
                question["expected_source"],
                outcome.rewritten_query,
                json.dumps(outcome.hits),
                outcome.expected_rank,
                outcome.expected_sent,
                outcome.answer,
                outcome.error,
                outcome.prompt_tokens,
                outcome.completion_tokens,
                outcome.ms,
            )
            await conn.execute(
                "update eval_batches set done = done + 1 where id = $1", uuid.UUID(batch_id)
            )

    async def finish_batch(self, batch_id: str, status: str, error: str | None) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "update eval_batches set status = $2, error = $3, finished_at = now() "
                "where id = $1",
                uuid.UUID(batch_id),
                status,
                error,
            )

    async def batches(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"select {BATCH_COLUMNS}, {SUMMARY_SQL} from eval_batches b "
                "left join eval_results r on r.batch_id = b.id "
                "group by b.id order by b.started_at desc limit 100"
            )
        return [_json(r) for r in rows]

    async def batch(self, batch_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"select {BATCH_COLUMNS}, {SUMMARY_SQL} from eval_batches b "
                "left join eval_results r on r.batch_id = b.id where b.id = $1 group by b.id",
                uuid.UUID(batch_id),
            )
        return _json(row)

    async def results(self, batch_id: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "select r.id, r.question_id, r.question_ta, q.question_en, r.expected_source, "
                "r.rewritten_query, r.hits::text as hits, r.expected_rank, r.expected_sent, "
                "r.answer, r.error, r.prompt_tokens, r.completion_tokens, r.ms, r.at "
                "from eval_results r left join eval_questions q on q.id = r.question_id "
                "where r.batch_id = $1 order by r.id",
                uuid.UUID(batch_id),
            )
        return [_json(r) for r in rows]

    async def delete_batch(self, batch_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "delete from eval_batches where id = $1", uuid.UUID(batch_id)
            )
        return result.endswith(" 1")


# ---------------------------------------------------------------------------
# Tamil conversion


async def to_tamil(
    api_key: str, model: str, text: str, *, reasoning_effort: str | None = None
) -> str:
    """One standalone, colloquial Tamil caller question for an English test question."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": TAMIL_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 200,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    async with httpx.AsyncClient(timeout=TRANSLATE_TIMEOUT_SECS) as client:
        response = await client.post(
            SARVAM_CHAT_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body
        )
    response.raise_for_status()
    data = response.json()
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    content = content.strip().strip('"').strip()
    if not content:
        raise ValueError("the model returned no Tamil text")
    return content


# ---------------------------------------------------------------------------
# workbook import


def read_workbook_questions(path: str) -> list[tuple[str, str]]:
    """``(question_en, sheet)`` for every example question in the review workbook."""
    from openpyxl import load_workbook

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(path)
    wb = load_workbook(file, read_only=True)
    out: list[tuple[str, str]] = []
    for name in wb.sheetnames:
        if not SHEET.match(name):
            continue
        for section, _item, content in wb[name].iter_rows(min_row=2, values_only=True):
            if section == "Sample question" and content and str(content).strip():
                out.append((str(content).strip(), name))
    return out


# ---------------------------------------------------------------------------
# the run


SessionFactory = Callable[[], ChatSession]
MAX_CONCURRENCY = 4
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECS = (3.0, 8.0, 20.0)


def transient_failure(turn: ChatTurn) -> bool:
    """The turn failed because Sarvam answered 429 or did not answer in time at some step:
    worth asking again, unlike a bad request."""
    if not turn.error:
        return False
    text = turn.error.lower()
    return "429" in text or "timed out" in text or "timeout" in text


class EvalRunner:
    """Runs batches in the background, one at a time, one question at a time."""

    def __init__(self, store: EvalStore) -> None:
        self.store = store
        self.task: asyncio.Task | None = None
        self.batch_id: str | None = None
        self._cancel = asyncio.Event()

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(
        self,
        batch_id: str,
        questions: list[dict],
        new_session: SessionFactory,
        close: Callable[[], Awaitable[None]] | None = None,
        concurrency: int = 1,
    ) -> None:
        if self.running:
            raise RuntimeError("a batch is already running")
        self.batch_id = batch_id
        self._cancel = asyncio.Event()
        workers = max(1, min(int(concurrency), MAX_CONCURRENCY))
        self.task = asyncio.get_running_loop().create_task(
            self._run(batch_id, questions, new_session, close, workers),
            name=f"eval-{batch_id[:8]}",
        )

    def cancel(self, batch_id: str) -> bool:
        if self.running and self.batch_id == batch_id:
            self._cancel.set()
            return True
        return False

    async def shutdown(self) -> None:
        if self.running:
            self._cancel.set()
            self.task.cancel()  # type: ignore[union-attr]
            await asyncio.gather(self.task, return_exceptions=True)  # type: ignore[arg-type]

    async def _ask(self, question: dict, new_session: SessionFactory) -> Outcome:
        """One question in a brand-new session; a 429 or a timeout from Sarvam is retried
        with backoff."""
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            session = new_session()  # cannot be built -> the whole batch fails
            try:
                turn = await session.say(question["question_ta"])
            except Exception as exc:  # one failed question must not stop the batch
                logger.warning(f"[EVAL] question {question['id']} failed: {exc!r}")
                return Outcome(None, [], None, False, "", str(exc)[:500], 0, 0, 0)
            if not transient_failure(turn) or attempt == RATE_LIMIT_RETRIES:
                return outcome_of(turn, question["expected_source"])
            wait = RATE_LIMIT_BACKOFF_SECS[min(attempt, len(RATE_LIMIT_BACKOFF_SECS) - 1)]
            logger.warning(f"[EVAL] q{question['id']} {turn.error[:60]!r}, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    async def _run(
        self,
        batch_id: str,
        questions: list[dict],
        new_session: SessionFactory,
        close: Callable[[], Awaitable[None]] | None,
        workers: int,
    ) -> None:
        status, error = "done", None
        started = time.monotonic()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        for question in questions:
            queue.put_nowait(question)
        done_count = 0

        async def worker() -> None:
            nonlocal done_count
            while not self._cancel.is_set():
                try:
                    question = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                outcome = await self._ask(question, new_session)
                await self.store.add_result(batch_id, question, outcome)
                done_count += 1
                logger.info(
                    f"[EVAL] {done_count}/{len(questions)} q{question['id']}: "
                    f"rank {outcome.expected_rank}, sent {outcome.expected_sent}, {outcome.ms} ms"
                )

        try:
            await asyncio.gather(*(worker() for _ in range(workers)))
            if self._cancel.is_set() and not queue.empty():
                status = "cancelled"
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status, error = "failed", str(exc)[:500]
            logger.exception("[EVAL] batch failed")
        finally:
            with_ms = round((time.monotonic() - started) * 1000)
            try:
                await self.store.finish_batch(batch_id, status, error)
            except Exception as exc:  # pragma: no cover - best effort at shutdown
                logger.warning(f"[EVAL] could not record batch end: {exc!r}")
            if close is not None:
                await close()
            logger.info(f"[EVAL] batch {batch_id[:8]} {status} after {with_ms} ms")


# ---------------------------------------------------------------------------
# routes


class QuestionIn(BaseModel):
    question_en: str
    question_ta: str | None = None
    expected_source: str
    note: str | None = None


class QuestionPatch(BaseModel):
    question_en: str | None = None
    question_ta: str | None = None
    expected_source: str | None = None
    note: str | None = None
    enabled: bool | None = None


class ImportIn(BaseModel):
    path: str


class BatchIn(BaseModel):
    note: str | None = None
    concurrency: int = 1
    search_type: SearchType = "semantic"
    """semantic (embeddings, as calls do), keyword (BM25) or hybrid (both, merged by RRF)."""
    collection: str | None = None
    """Qdrant collection to search instead of the configured one (a copy under test)."""
    question_ids: list[int] | None = None
    """Run only these question ids; empty or missing means every runnable question."""


class RetranslateIn(BaseModel):
    origin: str | None = "workbook"


def eval_settings_for(settings: Settings, config: AgentConfig) -> Settings:
    """Settings with the active config's language and model choices applied, as a call does."""
    from voice_agent.session import apply_language_override, apply_model_overrides

    return apply_model_overrides(apply_language_override(settings, config.language), config)


def create_evals_router(settings: Settings | None) -> APIRouter:
    router = APIRouter(prefix="/api/evals")
    runner: EvalRunner | None = None
    translating: asyncio.Task | None = None

    def agent_settings() -> Settings:
        if settings is None:
            raise HTTPException(503, "evaluation needs the Sarvam key in backend/.env")
        return settings

    def store(request: Request) -> PostgresEvalStore:
        return PostgresEvalStore(request.app.state.pool)

    def get_runner(request: Request) -> EvalRunner:
        nonlocal runner
        if runner is None:
            runner = EvalRunner(PostgresEvalStore(request.app.state.pool))
            request.app.state.eval_runner = runner
        return runner

    async def translator(request: Request):
        """A function that converts one English question to Tamil with the active model."""
        s = agent_settings()
        config = await load_active_config(request.app.state.pool)
        model = config.llm_model or s.llm_model
        effort = catalogue.reasoning_effort_for(model, s.llm_reasoning_effort)
        key = s.sarvam_api_key.get_secret_value()

        async def convert(text: str) -> str:
            return await to_tamil(key, model, text, reasoning_effort=effort)

        return convert

    async def translate_missing(request: Request) -> None:
        """Background: fill ``question_ta`` for rows that have none, one at a time."""
        convert = await translator(request)
        st = store(request)
        for row in await st.untranslated():
            try:
                tamil = await convert(row["question_en"])
            except Exception as exc:
                logger.warning(f"[EVAL] could not translate q{row['id']}: {exc!r}")
                continue
            await st.update_question(row["id"], question_ta=tamil)

    def start_translation(request: Request) -> None:
        nonlocal translating
        if translating is not None and not translating.done():
            return
        translating = asyncio.get_running_loop().create_task(translate_missing(request))

    # -- questions ----------------------------------------------------------

    @router.get("/questions")
    async def api_questions(
        request: Request, offset: int = 0, limit: int = 50, q: str | None = None
    ):
        st = store(request)
        rows, total = await st.questions(
            offset=max(0, offset), limit=max(1, min(limit, 200)), q=(q or "").strip() or None
        )
        return {
            "questions": rows,
            "total": total,
            "offset": max(0, offset),
            "limit": max(1, min(limit, 200)),
            "untranslated": len(await st.untranslated()),
            "runnable": len(await st.runnable_questions()),
            "translating": translating is not None and not translating.done(),
        }

    @router.post("/questions", status_code=201)
    async def api_add_question(request: Request, body: QuestionIn):
        en = body.question_en.strip()
        if not en or not body.expected_source.strip():
            raise HTTPException(422, "question and expected chunk are required")
        ta = (body.question_ta or "").strip() or None
        if ta is None:
            try:
                ta = await (await translator(request))(en)
            except Exception as exc:
                logger.warning(f"[EVAL] Tamil conversion failed: {exc!r}")
        return await store(request).add_question(
            en, ta, body.expected_source.strip(), body.note, "manual"
        )

    @router.put("/questions/{question_id}")
    async def api_update_question(request: Request, question_id: int, body: QuestionPatch):
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        for key in ("question_en", "question_ta", "expected_source"):
            if key in fields:
                fields[key] = fields[key].strip()
                if not fields[key] and key != "question_ta":
                    raise HTTPException(422, f"{key} cannot be empty")
        row = await store(request).update_question(question_id, **fields)
        if row is None:
            raise HTTPException(404, "question not found")
        return row

    @router.delete("/questions/{question_id}")
    async def api_delete_question(request: Request, question_id: int):
        return {"deleted": await store(request).delete_question(question_id)}

    @router.post("/questions/{question_id}/translate")
    async def api_translate_question(request: Request, question_id: int):
        row = await store(request).question(question_id)
        if row is None:
            raise HTTPException(404, "question not found")
        try:
            tamil = await (await translator(request))(row["question_en"])
        except Exception as exc:
            raise HTTPException(502, f"Tamil conversion failed: {exc}") from exc
        return await store(request).update_question(question_id, question_ta=tamil)

    @router.post("/questions/retranslate", status_code=202)
    async def api_retranslate(request: Request, body: RetranslateIn):
        """Write the Tamil again for the imported questions (or all), in the background."""
        agent_settings()
        cleared = await store(request).clear_tamil(body.origin)
        start_translation(request)
        return {"cleared": cleared, "started": True}

    @router.post("/questions/translate-missing", status_code=202)
    async def api_translate_missing(request: Request):
        agent_settings()
        start_translation(request)
        return {"started": True}

    @router.post("/questions/import")
    async def api_import(request: Request, body: ImportIn):
        """Load the review workbook's example questions; Tamil is filled in the background."""
        try:
            pairs = read_workbook_questions(body.path.strip())
        except FileNotFoundError:
            raise HTTPException(422, f"workbook not found: {body.path}") from None
        except Exception as exc:
            raise HTTPException(422, f"could not read the workbook: {exc}") from exc
        st = store(request)
        existing = await st.existing_keys()
        added = 0
        for question_en, sheet in pairs:
            if (question_en, sheet) in existing:
                continue
            await st.add_question(question_en, None, sheet, None, "workbook")
            existing.add((question_en, sheet))
            added += 1
        if settings is not None and added:
            start_translation(request)
        return {"read": len(pairs), "added": added, "skipped": len(pairs) - added}

    @router.get("/sources")
    async def api_sources(request: Request):
        """The sheet ids in the active collection, for the expected-chunk picker."""
        kstore = request.app.state.knowledge_store
        if kstore is None:
            return {"sources": []}
        try:
            names = await kstore.document_names()
        except Exception as exc:
            raise HTTPException(502, f"Qdrant: {exc}") from exc
        return {"sources": [n.split(" ", 1)[0] for n in names], "names": names}

    # -- batches ------------------------------------------------------------

    @router.post("/batches", status_code=201)
    async def api_run(request: Request, body: BatchIn):
        s = agent_settings()
        run = get_runner(request)
        if run.running:
            raise HTTPException(409, "a batch is already running; wait for it or cancel it")
        knowledge: KnowledgeService | None = request.app.state.knowledge
        if knowledge is None:
            raise HTTPException(503, "knowledge search is not available (OPENAI_API_KEY, Qdrant)")
        questions = await store(request).runnable_questions()
        if body.question_ids:
            wanted = set(body.question_ids)
            questions = [q for q in questions if q["id"] in wanted]
            missing = sorted(wanted - {q["id"] for q in questions})
            if missing:
                raise HTTPException(
                    422, f"questions not runnable (unknown, disabled or no Tamil): {missing}"
                )
        if not questions:
            raise HTTPException(422, "no enabled questions with a Tamil text to run")
        config = await load_active_config(request.app.state.pool)
        eval_settings = eval_settings_for(s, config)
        collection = (body.collection or "").strip()
        if collection:
            eval_settings = eval_settings.model_copy(update={"qdrant_collection": collection})
        from voice_agent.services import build_knowledge_service

        # the call's own knowledge service (translate/rewrite models as on a call),
        # searching the way this batch asks for
        eval_knowledge = (
            build_knowledge_service(eval_settings, search_type=body.search_type) or knowledge
        )
        own = eval_knowledge is not knowledge
        if not own and body.search_type != "semantic":
            raise HTTPException(
                400, f"{body.search_type} search needs the agent's own knowledge service"
            )
        if collection:
            if not own:
                raise HTTPException(
                    400, "another collection needs the agent's own knowledge service"
                )
            if not await eval_knowledge.collection_exists():
                await eval_knowledge.close()
                raise HTTPException(404, f"Qdrant collection {collection!r} does not exist")

        def new_session() -> ChatSession:
            # No LLM decide step: every question is searched as said, so each result
            # holds retrieved sheets instead of a counter-question.
            return ChatSession(
                settings=eval_settings,
                config=config,
                knowledge=eval_knowledge,
                force_search=True,
                llm_retries=0,  # the runner retries the whole question with backoff
            )

        async def close() -> None:
            if own:
                await eval_knowledge.close()

        workers = max(1, min(body.concurrency, MAX_CONCURRENCY))
        batch_id = str(uuid.uuid4())
        batch = await store(request).create_batch(
            batch_id,
            len(questions),
            config.id,
            eval_settings.qdrant_collection,
            (body.note or "").strip() or None,
            workers,
            force_search=True,
            search_type=body.search_type,
        )
        run.start(batch_id, questions, new_session, close, concurrency=workers)
        logger.info(
            f"[EVAL] batch {batch_id[:8]} started: {len(questions)} questions, "
            f"{workers} at a time, {body.search_type} search, config #{config.id}, "
            f"collection {eval_settings.qdrant_collection}"
        )
        return batch

    @router.get("/batches")
    async def api_batches(request: Request):
        run = get_runner(request)
        return {
            "batches": await store(request).batches(),
            "running": run.batch_id if run.running else None,
        }

    @router.get("/batches/{batch_id}")
    async def api_batch(request: Request, batch_id: str):
        st = store(request)
        try:
            batch = await st.batch(batch_id)
        except ValueError:
            raise HTTPException(404, "batch not found") from None
        if batch is None:
            raise HTTPException(404, "batch not found")
        return {"batch": batch, "results": await st.results(batch_id)}

    @router.post("/batches/{batch_id}/cancel")
    async def api_cancel(request: Request, batch_id: str):
        return {"cancelled": get_runner(request).cancel(batch_id)}

    @router.delete("/batches/{batch_id}")
    async def api_delete_batch(request: Request, batch_id: str):
        run = get_runner(request)
        if run.running and run.batch_id == batch_id:
            raise HTTPException(409, "cancel the batch before deleting it")
        try:
            return {"deleted": await store(request).delete_batch(batch_id)}
        except ValueError:
            raise HTTPException(404, "batch not found") from None

    return router
