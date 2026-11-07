"""Evaluation runs: outcome derivation, the background runner and the workbook reader."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chat_agent.session import ChatSession, ChatTurn
from portal.evals import (
    EvalRunner,
    Outcome,
    outcome_of,
    read_workbook_questions,
    source_matches,
)
from tests.test_chat import CONFIG, FakeKnowledge, FakeSarvam


def turn_with(searches: list[dict], reply: str = "reply", error: str | None = None) -> ChatTurn:
    return ChatTurn(
        index=1,
        user_text="q",
        reply=reply,
        searched=bool(searches),
        filler=None,
        topic=None,
        steps=[],
        searches=searches,
        dropped_text=None,
        prompt_tokens=10,
        completion_tokens=5,
        ms=120,
        error=error,
    )


def hit(source: str, score: float, used: bool) -> dict:
    return {"source": source, "document_id": "d", "chunk_index": 0, "score": score, "used": used}


# ---------- outcome ----------


def test_source_matches_on_the_sheet_id_only():
    assert source_matches("pb87-2 Commodity quantities by card type", "pb87-2")
    assert source_matches("pb87-2", "pb87-2")
    assert not source_matches("pb87-20 Something", "pb87-2")
    assert not source_matches("pb8 Other", "pb87")
    assert not source_matches(None, "pb87")


def test_outcome_rank_and_sent_flag_of_the_expected_chunk():
    turn = turn_with(
        [
            {
                "rewritten_query": "how much rice for 4 members",
                "hits": [
                    hit("pb140 SSR rice", 0.62, True),
                    hit("pb87-2 Commodity quantities", 0.58, True),
                    hit("pb89 Wheat", 0.30, False),
                ],
            }
        ]
    )
    out = outcome_of(turn, "pb87-2")
    assert out.expected_rank == 2 and out.expected_sent is True
    assert out.rewritten_query == "how much rice for 4 members"
    assert [h["source"] for h in out.hits] == [
        "pb140 SSR rice",
        "pb87-2 Commodity quantities",
        "pb89 Wheat",
    ]
    assert out.answer == "reply" and out.ms == 120 and out.prompt_tokens == 10


def test_outcome_when_the_expected_chunk_was_retrieved_but_not_sent():
    turn = turn_with(
        [{"hits": [hit("pb4 No records", 0.5, True), hit("pb37 Inactive", 0.3, False)]}]
    )
    out = outcome_of(turn, "pb37")
    assert out.expected_rank == 2 and out.expected_sent is False


def test_outcome_without_any_search_or_hit():
    assert outcome_of(turn_with([]), "pb1").expected_rank is None
    out = outcome_of(
        turn_with([{"translated_query": "x", "hits": [hit("pb2 Other", 0.5, True)]}]), "pb1"
    )
    assert out.expected_rank is None and out.expected_sent is False and out.rewritten_query == "x"


# ---------- the runner ----------


class MemoryStore:
    """In-memory stand-in for the Postgres eval store."""

    def __init__(self) -> None:
        self.results: list[tuple[str, int, Outcome]] = []
        self.finished: list[tuple[str, str, str | None]] = []
        self.done = 0

    async def add_result(self, batch_id: str, question: dict, outcome: Outcome) -> None:
        self.results.append((batch_id, question["id"], outcome))
        self.done += 1

    async def finish_batch(self, batch_id: str, status: str, error: str | None) -> None:
        self.finished.append((batch_id, status, error))


QUESTIONS = [
    {"id": 1, "question_ta": "ரேஷன் கார்டு எப்படி அப்ளை பண்றது?", "expected_source": "000 - Sheet.pdf"},
    {"id": 2, "question_ta": "கார்டு சரண்டர் பண்ணணும்", "expected_source": "999 - Missing.pdf"},
]


async def test_runner_asks_each_question_in_a_fresh_session_and_records_results(
    settings, monkeypatch
):
    # two questions, each: decide (search) then answer
    sarvam = FakeSarvam(
        {"questions": ["q1"]},
        {"text": "answer one"},
        {"questions": ["q2"]},
        {"text": "answer two"},
    )
    sarvam.install(monkeypatch)
    knowledge = FakeKnowledge(0.71, 0.55)
    sessions: list[ChatSession] = []

    def new_session() -> ChatSession:
        s = ChatSession(settings=settings, config=CONFIG, knowledge=knowledge)
        sessions.append(s)
        return s

    closed = []

    async def close() -> None:
        closed.append(True)

    store = MemoryStore()
    runner = EvalRunner(store)  # type: ignore[arg-type]
    runner.start("batch-1", QUESTIONS, new_session, close)
    assert runner.running
    await runner.task
    assert len(sessions) == 2 and all(len(s.turns) == 1 for s in sessions)
    assert [(b, q) for b, q, _ in store.results] == [("batch-1", 1), ("batch-1", 2)]
    first, second = (o for _, _, o in store.results)
    assert first.expected_rank == 1 and first.expected_sent and first.answer == "answer one"
    assert (
        second.expected_rank is None and not second.expected_sent and second.answer == "answer two"
    )
    assert store.finished == [("batch-1", "done", None)] and closed == [True]
    assert not runner.running


async def test_runner_records_a_failed_question_and_carries_on(settings, monkeypatch):
    sarvam = FakeSarvam({"text": "answer two"})  # only the second question gets a reply
    sarvam.install(monkeypatch)
    store = MemoryStore()
    calls = 0

    class Broken(ChatSession):
        async def say(self, text: str):
            raise RuntimeError("boom")

    def new_session() -> ChatSession:
        nonlocal calls
        calls += 1
        cls = Broken if calls == 1 else ChatSession
        return cls(settings=settings, config=CONFIG, knowledge=None)

    runner = EvalRunner(store)  # type: ignore[arg-type]
    runner.start("b", QUESTIONS, new_session)
    await runner.task
    first, second = (o for _, _, o in store.results)
    assert first.error == "boom" and first.answer == "" and first.expected_rank is None
    assert second.answer == "answer two" and second.error is None
    assert store.finished == [("b", "done", None)]


async def test_runner_marks_the_batch_failed_when_a_session_cannot_be_built(settings):
    store = MemoryStore()

    def new_session() -> ChatSession:
        raise RuntimeError("no knowledge")

    runner = EvalRunner(store)  # type: ignore[arg-type]
    runner.start("b", QUESTIONS, new_session)
    await runner.task
    assert store.results == [] and store.finished == [("b", "failed", "no knowledge")]


async def test_runner_cancel_stops_after_the_current_question(settings, monkeypatch):
    sarvam = FakeSarvam({"text": "one"}, {"text": "two"})
    sarvam.install(monkeypatch)
    store = MemoryStore()
    runner = EvalRunner(store)  # type: ignore[arg-type]
    started = asyncio.Event()
    release = asyncio.Event()

    class Slow(ChatSession):
        async def say(self, text: str):
            started.set()
            await release.wait()  # the first question is "in flight" until the test lets go
            return await super().say(text)

    def new_session() -> ChatSession:
        return Slow(settings=settings, config=CONFIG, knowledge=None)

    runner.start("b", QUESTIONS, new_session)
    await started.wait()
    assert runner.cancel("b")  # arrives while question 1 is still running
    release.set()
    await runner.task
    assert store.done == 1 and store.finished == [("b", "cancelled", None)]
    assert not runner.cancel("b")  # nothing running any more


def test_runner_refuses_a_second_batch_while_one_runs(settings):
    async def go():
        runner = EvalRunner(MemoryStore())  # type: ignore[arg-type]
        gate = asyncio.Event()

        def new_session() -> ChatSession:
            raise RuntimeError("never reached")

        async def slow_close():
            await gate.wait()

        runner.start("one", [], new_session, slow_close)
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            runner.start("two", [], new_session)
        gate.set()
        await runner.task

    asyncio.run(go())


# ---------- workbook ----------


def test_read_workbook_questions_takes_sample_rows_of_pb_sheets(tmp_path: Path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "pb87-2"
    ws.append(["Section", "Item", "Content"])
    ws.append(["Topic", "Quantities", "..."])
    ws.append(["Sample question", "1", "How much rice do I get?"])
    ws.append(["Sample question", "2", "  "])
    other = wb.create_sheet("Sheet2")
    other.append(["Sample question", "1", "ignored"])
    path = tmp_path / "wb.xlsx"
    wb.save(path)
    assert read_workbook_questions(str(path)) == [("How much rice do I get?", "pb87-2")]
    with pytest.raises(FileNotFoundError):
        read_workbook_questions(str(tmp_path / "missing.xlsx"))


# ---------- concurrency and rate limits ----------


async def test_runner_runs_the_requested_number_of_questions_at_a_time(settings):
    store = MemoryStore()
    runner = EvalRunner(store)  # type: ignore[arg-type]
    in_flight = 0
    peak = 0
    release = asyncio.Event()

    class Slow(ChatSession):
        async def say(self, text: str):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await release.wait()
            in_flight -= 1
            return turn_with([], reply="ok")

    questions = [dict(QUESTIONS[0], id=i) for i in range(1, 6)]
    runner.start("b", questions, lambda: Slow(settings=settings, config=CONFIG), concurrency=3)
    for _ in range(10):  # let the workers pick up their first questions
        await asyncio.sleep(0)
    assert in_flight == 3
    release.set()
    await runner.task
    assert peak == 3 and store.done == 5 and store.finished == [("b", "done", None)]


async def test_runner_retries_a_question_that_was_rate_limited(settings, monkeypatch):
    monkeypatch.setattr("portal.evals.RATE_LIMIT_BACKOFF_SECS", (0.0, 0.0, 0.0))
    store = MemoryStore()
    runner = EvalRunner(store)  # type: ignore[arg-type]
    attempts = 0

    class Flaky(ChatSession):
        async def say(self, text: str):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return turn_with(
                    [], reply="", error="llm decide failed: Client error '429 Too Many'"
                )
            return turn_with([{"hits": [hit("000 - Sheet.pdf", 0.9, True)]}], reply="finally")

    runner.start("b", QUESTIONS[:1], lambda: Flaky(settings=settings, config=CONFIG))
    await runner.task
    assert attempts == 3
    (_, _, outcome) = store.results[0]
    assert outcome.answer == "finally" and outcome.expected_rank == 1 and outcome.error is None


async def test_runner_gives_up_after_the_retry_limit(settings, monkeypatch):
    monkeypatch.setattr("portal.evals.RATE_LIMIT_BACKOFF_SECS", (0.0,))
    store = MemoryStore()
    runner = EvalRunner(store)  # type: ignore[arg-type]

    class Always429(ChatSession):
        async def say(self, text: str):
            return turn_with([], reply="", error="llm answer failed: 429 Too Many Requests")

    runner.start("b", QUESTIONS[:1], lambda: Always429(settings=settings, config=CONFIG))
    await runner.task
    (_, _, outcome) = store.results[0]
    assert "429" in (outcome.error or "") and store.finished == [("b", "done", None)]


# ---------- search type ----------


def test_batch_request_accepts_only_the_three_search_types():
    from pydantic import ValidationError

    from portal.evals import BatchIn

    assert BatchIn().search_type == "semantic"
    assert BatchIn(search_type="hybrid").search_type == "hybrid"
    with pytest.raises(ValidationError):
        BatchIn(search_type="bm25")  # type: ignore[arg-type]


def test_voice_and_chat_builders_follow_the_search_type_setting(settings):
    from voice_agent.services import build_knowledge_service

    s = settings.model_copy(update={"knowledge_enabled": True})
    if s.openai_api_key is None:
        from pydantic import SecretStr

        s = s.model_copy(update={"openai_api_key": SecretStr("sk-test")})
    service = build_knowledge_service(s)
    assert service is not None and service.search_type == "hybrid"  # KNOWLEDGE_SEARCH_TYPE
    assert build_knowledge_service(s, search_type="keyword").search_type == "keyword"


async def test_runner_retries_a_question_whose_answer_timed_out(settings, monkeypatch):
    monkeypatch.setattr("portal.evals.RATE_LIMIT_BACKOFF_SECS", (0.0,))
    store = MemoryStore()
    runner = EvalRunner(store)  # type: ignore[arg-type]
    attempts = 0

    class SlowOnce(ChatSession):
        async def say(self, text: str):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return turn_with([], reply="", error="llm answer failed: timed out after 90 s")
            return turn_with([{"hits": [hit("000 - Sheet.pdf", 0.9, True)]}], reply="late")

    runner.start("b", QUESTIONS[:1], lambda: SlowOnce(settings=settings, config=CONFIG))
    await runner.task
    (_, _, outcome) = store.results[0]
    assert attempts == 2 and outcome.answer == "late" and outcome.error is None


def test_transient_failures_are_429_and_timeouts_only():
    from portal.evals import transient_failure

    assert transient_failure(turn_with([], error="llm decide failed: 429 Too Many Requests"))
    assert transient_failure(turn_with([], error="llm answer failed: timed out after 90 s"))
    assert not transient_failure(turn_with([], error="llm answer failed: 400 Bad Request"))
    assert not transient_failure(turn_with([], error=None))
