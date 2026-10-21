"""Queue-backed writer that persists calls and turns without blocking the pipeline."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import asyncpg
from loguru import logger

WriteOp = Callable[[asyncpg.Connection], Awaitable[None]]


@dataclass(frozen=True)
class CallInfo:
    """Static facts about a session, known before the pipeline starts."""

    transport: str
    language: str
    session_id: str | None = None
    direction: str | None = None
    from_number: str | None = None
    to_number: str | None = None
    call_id: str | None = None
    stream_id: str | None = None
    config_id: int | None = None


@dataclass
class TurnRecord:
    """One user/assistant exchange with its latency measurements."""

    turn_index: int
    user_text: str | None = None
    assistant_text: str | None = None
    user_spoke_at: datetime | None = None
    bot_spoke_at: datetime | None = None
    completed_at: datetime | None = None
    response_latency_ms: int | None = None
    turn_release_ms: int | None = None
    stt_ttfb_ms: int | None = None
    llm_ttfb_ms: int | None = None
    tts_ttfb_ms: int | None = None
    interrupted: bool = False
    contributions: list[dict[str, object]] = field(default_factory=list)
    """Pipecat's non-overlapping parts of the latency interval, in order."""
    searches: list[dict[str, object]] = field(default_factory=list)
    """Knowledge-base searches made while answering this turn, with retrieved chunks."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tts_characters: int | None = None
    stt_audio_seconds: float | None = None


@dataclass
class _Stats:
    queued: int = 0
    written: int = 0
    dropped: int = 0
    failed: int = 0
    max_write_ms: int = 0
    """Slowest single database write, measured in the background task."""


class CallRecorder:
    """Persists one call. All public methods return immediately except ``close``.

    Writes are serialised through an ``asyncio.Queue`` drained by a background
    task; if the queue is full (database stalled) new records are dropped with
    a warning rather than slowing the conversation.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        call: CallInfo,
        *,
        call_id: uuid.UUID | None = None,
        max_queue: int = 500,
        flush_timeout: float = 5.0,
    ) -> None:
        self._pool = pool
        self.call = call
        self.id = call_id or uuid.uuid4()
        self.started_at = datetime.now(UTC)
        self._queue: asyncio.Queue[WriteOp] = asyncio.Queue(maxsize=max_queue)
        self._writer: asyncio.Task | None = None
        self._flush_timeout = flush_timeout
        self.stats = _Stats()

    def start(self) -> None:
        self._writer = asyncio.create_task(self._drain(), name=f"call-recorder-{self.id}")
        self._enqueue(self._insert_call)

    def turn_completed(self, turn: TurnRecord) -> None:
        self._enqueue(lambda conn: self._insert_turn(conn, turn))

    async def close(self, end_reason: str) -> None:
        """Record the end of the call and wait briefly for pending writes."""
        ended_at = datetime.now(UTC)
        self._enqueue(lambda conn: self._finish_call(conn, ended_at, end_reason))
        try:
            await asyncio.wait_for(self._queue.join(), timeout=self._flush_timeout)
        except TimeoutError:
            logger.warning(f"Call {self.id}: {self._queue.qsize()} tracking writes not flushed")
        if self._writer:
            self._writer.cancel()
            await asyncio.gather(self._writer, return_exceptions=True)
        logger.info(f"Call {self.id} tracked: {self.stats}")

    # -- internals ---------------------------------------------------------

    def _enqueue(self, op: WriteOp) -> None:
        try:
            self._queue.put_nowait(op)
            self.stats.queued += 1
        except asyncio.QueueFull:
            self.stats.dropped += 1
            if self.stats.dropped == 1:
                logger.warning(f"Call {self.id}: tracking queue full, dropping records")

    async def _drain(self) -> None:
        while True:
            op = await self._queue.get()
            started = time.monotonic()
            try:
                async with self._pool.acquire() as conn:
                    await op(conn)
                self.stats.written += 1
                elapsed_ms = round((time.monotonic() - started) * 1000)
                self.stats.max_write_ms = max(self.stats.max_write_ms, elapsed_ms)
            except Exception as exc:
                self.stats.failed += 1
                if self.stats.failed == 1:
                    logger.warning(f"Call {self.id}: tracking write failed: {exc}")
            finally:
                self._queue.task_done()

    async def _insert_call(self, conn: asyncpg.Connection) -> None:
        c = self.call
        await conn.execute(
            "insert into calls (id, session_id, transport, direction, from_number, to_number, "
            "call_id, stream_id, language, config_id, started_at) "
            "values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
            self.id,
            c.session_id,
            c.transport,
            c.direction,
            c.from_number,
            c.to_number,
            c.call_id,
            c.stream_id,
            c.language,
            c.config_id,
            self.started_at,
        )

    async def _insert_turn(self, conn: asyncpg.Connection, t: TurnRecord) -> None:
        await conn.execute(
            "insert into turns (call_id, turn_index, user_text, assistant_text, user_spoke_at, "
            "bot_spoke_at, completed_at, response_latency_ms, turn_release_ms, stt_ttfb_ms, "
            "llm_ttfb_ms, tts_ttfb_ms, interrupted, contributions, searches, "
            "prompt_tokens, completion_tokens, tts_characters, stt_audio_seconds) "
            "values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, "
            "$15::jsonb, $16, $17, $18, $19) "
            "on conflict (call_id, turn_index) do nothing",
            self.id,
            t.turn_index,
            t.user_text,
            t.assistant_text,
            t.user_spoke_at,
            t.bot_spoke_at,
            t.completed_at,
            t.response_latency_ms,
            t.turn_release_ms,
            t.stt_ttfb_ms,
            t.llm_ttfb_ms,
            t.tts_ttfb_ms,
            t.interrupted,
            json.dumps(t.contributions),
            json.dumps(t.searches),
            t.prompt_tokens,
            t.completion_tokens,
            t.tts_characters,
            t.stt_audio_seconds,
        )

    async def _finish_call(self, conn: asyncpg.Connection, ended_at: datetime, reason: str) -> None:
        await conn.execute(
            "update calls set ended_at = $2, end_reason = $3, "
            "turn_count = (select count(*) from turns where call_id = $1) where id = $1",
            self.id,
            ended_at,
            reason,
        )
