"""Answer-prompt evaluation: a fixed set of conversations, each with the sheets a search
returned, answered with one answer prompt (LLM 2).

The answer step writes the caller's reply from search results. To improve its prompt the
input has to stay fixed: the same conversations (single questions and follow-ups taken
from real calls) with the same sheets, frozen when the item was created, so two runs
differ only in the prompt and the model. Nothing here changes the saved configuration.

Items belong to a named dataset (``main`` by default); a run answers one dataset.

Mounted by the portal under ``/api/answer-evals``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import asyncpg
from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from chat_agent.session import ChatSession, error_text
from portal.evals import _json, eval_settings_for
from voice_agent.config import Settings
from voice_agent.knowledge.tool import ANSWER_PROMPT, TOOL_NAME, KnowledgeTool
from voice_agent.tracking.config_store import AgentConfig, load_active_config

CONCURRENCY = 2
TOOL_CALL_ID = "answer-eval"
DEFAULT_DATASET = "main"


def dataset_name(value: str | None) -> str:
    return " ".join((value or "").split())[:60] or DEFAULT_DATASET


@dataclass(frozen=True)
class AnswerOutcome:
    answer: str | None
    prompt_tokens: int
    completion_tokens: int
    ms: int
    error: str | None


def last_user_text(messages: list[dict]) -> str:
    """The caller's latest message, the one the answer is for."""
    for m in reversed(messages):
        if m.get("role") == "user" and (m.get("text") or "").strip():
            return m["text"].strip()
    return ""


def caller_messages(messages: list[dict]) -> list[str]:
    return [m["text"].strip() for m in messages if m.get("role") == "user" and m.get("text")]


def tool_result(item: dict, max_sheets: int | None = None, min_score: float | None = None) -> dict:
    """The search_knowledge_base result exactly as the call's tool hands it to the LLM:
    only the sheets marked ``used`` are passed, as passages. ``max_sheets`` (a smaller top-k)
    and ``min_score`` (a score floor; the best sheet always stays) try other selection rules
    without re-running the search."""
    question = last_user_text(item["messages"])
    sheets = [s for s in item.get("sheets") or [] if s.get("used", True)]
    if min_score is not None and sheets:
        sheets = [s for s in sheets if float(s.get("score") or 0) >= min_score] or sheets[:1]
    if max_sheets:
        sheets = sheets[:max_sheets]
    per_question = {
        "question": question,
        "query_in_english": item.get("rewritten_query"),
        "found": bool(sheets),
        "sources": [s["source"] for s in sheets],
    }
    if not sheets:
        return {"found": False, "reason": "no relevant information", "questions": [per_question]}
    passages = [
        {"source": s["source"], "score": round(float(s.get("score") or 0), 2), "text": s["text"]}
        for s in sheets
    ]
    return {"found": True, "questions": [per_question], "passages": passages}


def answer_messages(
    item: dict, system: str, max_sheets: int | None = None, min_score: float | None = None
) -> list[dict]:
    """The chat-completion request for the answer step: persona + answer rules, the
    conversation so far, the model's search call and its result, as on a call."""
    question = last_user_text(item["messages"])
    out: list[dict] = [{"role": "system", "content": system}]
    for m in item["messages"]:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        out.append({"role": "user" if m.get("role") == "user" else "assistant", "content": text})
    out.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": TOOL_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "arguments": json.dumps({"questions": [question]}, ensure_ascii=False),
                    },
                }
            ],
        }
    )
    out.append(
        {
            "role": "tool",
            "tool_call_id": TOOL_CALL_ID,
            "name": TOOL_NAME,
            "content": json.dumps(tool_result(item, max_sheets, min_score), ensure_ascii=False),
        }
    )
    return out


def clean_messages(messages: list[Any]) -> list[dict]:
    """The conversation as stored: role + trimmed text, blank messages dropped."""
    return [{"role": m.role, "text": m.text.strip()} for m in messages if m.text.strip()]


def prompt_source_of(prompt: str, config: AgentConfig) -> str:
    configured = (config.answer_prompt or "").strip()
    if prompt.strip() == ANSWER_PROMPT.strip():
        return "default"
    return "config" if prompt.strip() == configured else "draft"


async def answer_item(
    item: dict,
    *,
    session: Any,
    system: str,
    max_sheets: int | None = None,
    min_score: float | None = None,
) -> AnswerOutcome:
    """One answer for ``item`` with the session's model; ``system`` is persona + rules."""
    started = time.monotonic()
    try:
        reply = await session._complete(
            answer_messages(item, system, max_sheets, min_score), tools=None
        )
    except Exception as exc:
        ms = round((time.monotonic() - started) * 1000)
        return AnswerOutcome(None, 0, 0, ms, error_text(exc)[:300])
    return AnswerOutcome(
        reply.text or "",
        reply.prompt_tokens,
        reply.completion_tokens,
        round((time.monotonic() - started) * 1000),
        None if reply.text.strip() else "empty reply",
    )


async def fetch_sheets(
    messages: list[dict], *, service: Any, rewrite_prompt: str | None, limit: int, min_score: float
) -> tuple[list[dict], str | None]:
    """Search for the conversation's latest question the way a call does (rewrite of the
    caller's messages when the rewrite is on, else the question itself) and return the
    sheets with their text, marked ``used`` as the tool would select them."""
    caller = caller_messages(messages)
    if not caller:
        return [], None
    if getattr(service, "query_rewrite", False):
        result = await service.search_rewritten(caller, prompt=rewrite_prompt, limit=limit)
        if result is None:
            result = await service.search(caller[-1], limit=limit)
    else:
        result = await service.search(caller[-1], limit=limit)
    selector = KnowledgeTool(service=service, limit=limit, min_score=min_score)
    chosen = {id(h) for h in selector._selected(result.hits)}
    sheets = [
        {
            "source": h.document_name,
            "document_id": h.document_id,
            "chunk_index": h.chunk_index,
            "score": round(h.score, 3),
            "used": id(h) in chosen,
            "text": h.text,
        }
        for h in result.hits
    ]
    return sheets, result.rewritten_query or result.translated_query


# ---------- storage ----------

ITEM_COLUMNS = (
    "id, kind, messages, question_en, expected_source, sheets, rewritten_query, "
    "source_call_id, source_question_id, note, dataset, created_at, updated_at"
)
RUN_COLUMNS = (
    "r.id, r.started_at, r.finished_at, r.status, r.prompt, r.prompt_source, r.config_id, "
    "r.model, r.total, r.error, r.max_sheets, r.min_score, r.dataset"
)
RUN_SUMMARY_SQL = """
    count(x.id)                                   as results,
    count(x.id) filter (where x.error is null)    as answered,
    count(x.error)                                as errors,
    count(x.id) filter (where x.judged_ok)        as judged_ok,
    count(x.id) filter (where x.judged)           as judged,
    coalesce(round(avg(x.ms))::int, 0)            as avg_ms
"""


class PostgresAnswerStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def items(self, dataset: str | None = None) -> list[dict]:
        where = "where dataset = $1" if dataset else ""
        args = [dataset] if dataset else []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"select {ITEM_COLUMNS} from answer_eval_items {where} order by id", *args
            )
        return [_json(r) for r in rows]  # type: ignore[misc]

    async def datasets(self) -> list[dict]:
        """Every dataset name with its item count (runs without items still count)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "select dataset, count(*) as items from answer_eval_items group by dataset "
                "union all select distinct dataset, 0 from answer_eval_runs r "
                "where not exists (select 1 from answer_eval_items i where i.dataset = r.dataset) "
                "order by 1"
            )
        return [dict(r) for r in rows]

    async def item(self, item_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"select {ITEM_COLUMNS} from answer_eval_items where id = $1", item_id
            )
        return _json(row)

    async def add_item(
        self,
        *,
        kind: str,
        messages: list[dict],
        question_en: str | None,
        expected_source: str | None,
        sheets: list[dict],
        rewritten_query: str | None,
        source_call_id: str | None = None,
        source_question_id: int | None = None,
        note: str | None = None,
        dataset: str = DEFAULT_DATASET,
    ) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "insert into answer_eval_items (kind, messages, question_en, expected_source, "
                "sheets, rewritten_query, source_call_id, source_question_id, note, dataset) "
                "values ($1, $2::jsonb, $3, $4, $5::jsonb, $6, $7, $8, $9, $10) "
                f"returning {ITEM_COLUMNS}",
                kind,
                json.dumps(messages, ensure_ascii=False),
                question_en,
                expected_source,
                json.dumps(sheets, ensure_ascii=False),
                rewritten_query,
                uuid.UUID(source_call_id) if source_call_id else None,
                source_question_id,
                note,
                dataset,
            )
        return _json(row)  # type: ignore[return-value]

    async def update_item(
        self,
        item_id: int,
        *,
        messages: list[dict] | None = None,
        question_en: str | None = None,
        expected_source: str | None = None,
        sheets: list[dict] | None = None,
        rewritten_query: str | None = None,
        note: str | None = None,
        dataset: str | None = None,
    ) -> dict | None:
        sets, args = ["updated_at = now()"], []
        for col, val, is_json in (
            ("messages", messages, True),
            ("question_en", question_en, False),
            ("expected_source", expected_source, False),
            ("sheets", sheets, True),
            ("rewritten_query", rewritten_query, False),
            ("note", note, False),
            ("dataset", dataset, False),
        ):
            if val is None:
                continue
            args.append(json.dumps(val, ensure_ascii=False) if is_json else val)
            sets.append(f"{col} = ${len(args) + 1}" + ("::jsonb" if is_json else ""))
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"update answer_eval_items set {', '.join(sets)} where id = $1 "
                f"returning {ITEM_COLUMNS}",
                item_id,
                *args,
            )
        return _json(row)

    async def delete_item(self, item_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute("delete from answer_eval_items where id = $1", item_id)
        return result.endswith(" 1")

    async def random_bank_questions(self, count: int, dataset: str) -> list[dict]:
        """Enabled bank questions with Tamil, not yet in the dataset, at random."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "select q.id, q.question_ta, q.question_en, q.expected_source "
                "from eval_questions q where q.enabled "
                "and q.question_ta is not null and q.question_ta <> '' "
                "and q.id not in (select source_question_id from answer_eval_items "
                "where source_question_id is not null and dataset = $2) "
                "order by random() limit $1",
                count,
                dataset,
            )
        return [dict(r) for r in rows]

    async def random_follow_ups(self, count: int, dataset: str) -> list[dict]:
        """Follow-up turns from real calls (a user turn after the first that searched), one
        per call, with the earlier turns of that call as the conversation."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                with firsts as (
                    select call_id, min(turn_index) as first_turn from turns
                    where user_text is not null and user_text <> '' group by call_id
                ), candidates as (
                    select distinct on (t.call_id) t.call_id, t.turn_index, t.user_text
                    from turns t join firsts f on f.call_id = t.call_id
                    where t.user_text is not null and t.user_text <> ''
                      and t.turn_index > f.first_turn
                      and t.searches is not null and t.searches::text <> '[]'
                      and t.call_id not in (select source_call_id from answer_eval_items
                                            where source_call_id is not null and dataset = $2)
                    order by t.call_id, random()
                )
                select * from candidates order by random() limit $1
                """,
                count,
                dataset,
            )
            out = []
            for r in rows:
                earlier = await conn.fetch(
                    "select user_text, assistant_text from turns where call_id = $1 "
                    "and turn_index < $2 order by turn_index",
                    r["call_id"],
                    r["turn_index"],
                )
                messages: list[dict] = []
                for e in earlier:
                    if (e["user_text"] or "").strip():
                        messages.append({"role": "user", "text": e["user_text"].strip()})
                    if (e["assistant_text"] or "").strip():
                        messages.append({"role": "assistant", "text": e["assistant_text"].strip()})
                messages.append({"role": "user", "text": r["user_text"].strip()})
                out.append(
                    {
                        "call_id": str(r["call_id"]),
                        "turn_index": r["turn_index"],
                        "messages": messages,
                    }
                )
        return out

    async def create_run(
        self,
        run_id: str,
        *,
        prompt: str,
        prompt_source: str,
        config_id: int | None,
        model: str,
        total: int,
        max_sheets: int | None = None,
        min_score: float | None = None,
        dataset: str = DEFAULT_DATASET,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "insert into answer_eval_runs (id, status, prompt, prompt_source, config_id, "
                "model, total, max_sheets, min_score, dataset) "
                "values ($1, 'running', $2, $3, $4, $5, $6, $7, $8, $9)",
                uuid.UUID(run_id),
                prompt,
                prompt_source,
                config_id,
                model,
                total,
                max_sheets,
                min_score,
                dataset,
            )

    async def add_result(self, run_id: str, item: dict, outcome: AnswerOutcome) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "insert into answer_eval_results (run_id, item_id, answer, prompt_tokens, "
                "completion_tokens, ms, error) values ($1, $2, $3, $4, $5, $6, $7)",
                uuid.UUID(run_id),
                item["id"],
                outcome.answer,
                outcome.prompt_tokens,
                outcome.completion_tokens,
                outcome.ms,
                outcome.error,
            )

    async def finish_run(self, run_id: str, status: str, error: str | None) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "update answer_eval_runs set status = $2, error = $3, finished_at = now() "
                "where id = $1",
                uuid.UUID(run_id),
                status,
                error,
            )

    async def runs(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"select {RUN_COLUMNS}, {RUN_SUMMARY_SQL} from answer_eval_runs r "
                "left join answer_eval_results x on x.run_id = r.id "
                "group by r.id order by r.started_at desc limit 100"
            )
        return [_json(r) for r in rows]  # type: ignore[misc]

    async def run(self, run_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"select {RUN_COLUMNS}, {RUN_SUMMARY_SQL} from answer_eval_runs r "
                "left join answer_eval_results x on x.run_id = r.id where r.id = $1 group by r.id",
                uuid.UUID(run_id),
            )
        return _json(row)

    async def results(self, run_id: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "select id, run_id, item_id, answer, prompt_tokens, completion_tokens, ms, error, "
                "judged, judged_ok, judge_note, at from answer_eval_results where run_id = $1 "
                "order by item_id",
                uuid.UUID(run_id),
            )
        return [_json(r) for r in rows]  # type: ignore[misc]

    async def judge(self, run_id: str, result_id: int, ok: bool | None, note: str | None) -> bool:
        """The reviewer's verdict on one answer: ok True/False (None = not judged yet)."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "update answer_eval_results set judged = $3, judged_ok = $4, judge_note = $5 "
                "where id = $2 and run_id = $1",
                uuid.UUID(run_id),
                result_id,
                ok is not None,
                bool(ok),
                note,
            )
        return result.endswith(" 1")

    async def delete_run(self, run_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "delete from answer_eval_runs where id = $1", uuid.UUID(run_id)
            )
        return result.endswith(" 1")


# ---------- routes ----------


class MessageIn(BaseModel):
    role: str
    text: str


class ItemIn(BaseModel):
    messages: list[MessageIn]
    question_en: str | None = None
    expected_source: str | None = None
    note: str | None = None
    dataset: str | None = None
    """The dataset the item belongs to (default ``main``); a new name makes a new dataset."""


class GenerateIn(BaseModel):
    singles: int = 60
    """Single questions picked at random from the evaluation bank."""
    follow_ups: int = 40
    """Follow-up turns taken from real calls, with the earlier turns as context."""
    dataset: str | None = None


class JudgeIn(BaseModel):
    ok: bool | None
    """True = correct answer, False = wrong, None = clear the verdict."""
    note: str | None = None


class RunIn(BaseModel):
    prompt: str | None = None
    """The answer rules to try (the part after the persona); empty = the configuration's."""
    max_sheets: int | None = None
    """Pass only the first N sent sheets to the model (try a smaller top-k); None = all."""
    min_score: float | None = None
    """Pass only sent sheets scoring at least this (the best one always stays); None = all."""
    dataset: str | None = None
    """The dataset to answer (default ``main``)."""


def create_answer_evals_router(settings: Settings | None) -> APIRouter:
    router = APIRouter(prefix="/api/answer-evals")
    running = False

    def store(request: Request) -> PostgresAnswerStore:
        return PostgresAnswerStore(request.app.state.pool)

    def agent_settings() -> Settings:
        if settings is None:
            raise HTTPException(503, "evaluation needs the Sarvam key in backend/.env")
        return settings

    async def searcher(request: Request):
        """The call's own knowledge service (rewrite on when the config says so)."""
        s = agent_settings()
        if request.app.state.knowledge is None:
            raise HTTPException(503, "knowledge search is not available (OPENAI_API_KEY, Qdrant)")
        config = await load_active_config(request.app.state.pool)
        eval_settings = eval_settings_for(s, config)
        from voice_agent.services import build_knowledge_service

        service = build_knowledge_service(eval_settings)
        if service is None:
            raise HTTPException(503, "knowledge search is not available (OPENAI_API_KEY, Qdrant)")
        return service, config, eval_settings

    @router.get("/datasets")
    async def api_datasets(request: Request):
        return {"datasets": await store(request).datasets()}

    @router.get("/items")
    async def api_items(request: Request, dataset: str | None = None):
        return {"items": await store(request).items(dataset_name(dataset) if dataset else None)}

    @router.post("/items/generate", status_code=201)
    async def api_generate(request: Request, body: GenerateIn):
        st = store(request)
        dataset = dataset_name(body.dataset)
        service, config, es = await searcher(request)
        singles = await st.random_bank_questions(max(0, min(body.singles, 200)), dataset)
        follow_ups = await st.random_follow_ups(max(0, min(body.follow_ups, 200)), dataset)
        added = {"single": 0, "follow_up": 0}
        gate = asyncio.Semaphore(CONCURRENCY)

        async def add_single(q: dict) -> None:
            messages = [{"role": "user", "text": q["question_ta"]}]
            async with gate:
                sheets, query = await fetch_sheets(
                    messages,
                    service=service,
                    rewrite_prompt=config.rewrite_prompt,
                    limit=es.knowledge_top_k,
                    min_score=es.knowledge_min_score,
                )
            await st.add_item(
                kind="single",
                messages=messages,
                question_en=q["question_en"],
                expected_source=q["expected_source"],
                sheets=sheets,
                rewritten_query=query,
                source_question_id=q["id"],
                dataset=dataset,
            )
            added["single"] += 1

        async def add_follow_up(f: dict) -> None:
            async with gate:
                sheets, query = await fetch_sheets(
                    f["messages"],
                    service=service,
                    rewrite_prompt=config.rewrite_prompt,
                    limit=es.knowledge_top_k,
                    min_score=es.knowledge_min_score,
                )
            await st.add_item(
                kind="follow_up",
                messages=f["messages"],
                question_en=None,
                expected_source=None,
                sheets=sheets,
                rewritten_query=query,
                source_call_id=f["call_id"],
                note=f"call {f['call_id'][:8]} turn {f['turn_index']}",
                dataset=dataset,
            )
            added["follow_up"] += 1

        try:
            await asyncio.gather(
                *(add_single(q) for q in singles), *(add_follow_up(f) for f in follow_ups)
            )
        finally:
            await service.close()
        return {"added": added, "items": await st.items(dataset)}

    @router.post("/items", status_code=201)
    async def api_add_item(request: Request, body: ItemIn):
        messages = [
            {"role": m.role, "text": m.text.strip()} for m in body.messages if m.text.strip()
        ]
        if not messages or messages[-1]["role"] != "user":
            raise HTTPException(422, "the conversation must end with a caller message")
        service, config, es = await searcher(request)
        try:
            sheets, query = await fetch_sheets(
                messages,
                service=service,
                rewrite_prompt=config.rewrite_prompt,
                limit=es.knowledge_top_k,
                min_score=es.knowledge_min_score,
            )
        finally:
            await service.close()
        kind = "follow_up" if sum(1 for m in messages if m["role"] == "user") > 1 else "single"
        return await store(request).add_item(
            kind=kind,
            messages=messages,
            question_en=(body.question_en or "").strip() or None,
            expected_source=(body.expected_source or "").strip() or None,
            sheets=sheets,
            rewritten_query=query,
            note=(body.note or "").strip() or None,
            dataset=dataset_name(body.dataset),
        )

    @router.put("/items/{item_id}")
    async def api_update_item(request: Request, item_id: int, body: ItemIn):
        messages = [
            {"role": m.role, "text": m.text.strip()} for m in body.messages if m.text.strip()
        ]
        if not messages or messages[-1]["role"] != "user":
            raise HTTPException(422, "the conversation must end with a caller message")
        st = store(request)
        current = await st.item(item_id)
        if current is None:
            raise HTTPException(404, "item not found")
        sheets = query = None
        if messages != current["messages"]:  # the question changed: fetch its sheets again
            service, config, es = await searcher(request)
            try:
                sheets, query = await fetch_sheets(
                    messages,
                    service=service,
                    rewrite_prompt=config.rewrite_prompt,
                    limit=es.knowledge_top_k,
                    min_score=es.knowledge_min_score,
                )
            finally:
                await service.close()
        row = await st.update_item(
            item_id,
            messages=messages,
            question_en=(body.question_en or "").strip() or "",
            expected_source=(body.expected_source or "").strip() or "",
            sheets=sheets,
            rewritten_query=query,
            note=(body.note or "").strip() or "",
            dataset=dataset_name(body.dataset) if body.dataset else None,
        )
        return row

    @router.post("/items/{item_id}/refresh")
    async def api_refresh_item(request: Request, item_id: int):
        """Fetch the item's sheets again from the current collection."""
        st = store(request)
        current = await st.item(item_id)
        if current is None:
            raise HTTPException(404, "item not found")
        service, config, es = await searcher(request)
        try:
            sheets, query = await fetch_sheets(
                current["messages"],
                service=service,
                rewrite_prompt=config.rewrite_prompt,
                limit=es.knowledge_top_k,
                min_score=es.knowledge_min_score,
            )
        finally:
            await service.close()
        return await st.update_item(item_id, sheets=sheets, rewritten_query=query or "")

    @router.delete("/items/{item_id}")
    async def api_delete_item(request: Request, item_id: int):
        if not await store(request).delete_item(item_id):
            raise HTTPException(404, "item not found")
        return {"deleted": True}

    @router.post("/runs", status_code=201)
    async def api_run(request: Request, body: RunIn):
        nonlocal running
        if running:
            raise HTTPException(409, "an answer run is already in progress")
        s = agent_settings()
        st = store(request)
        dataset = dataset_name(body.dataset)
        items = await st.items(dataset)
        if not items:
            raise HTTPException(422, f"no items in dataset {dataset!r}: generate or add some first")
        config = await load_active_config(request.app.state.pool)
        eval_settings = eval_settings_for(s, config)
        draft = (body.prompt or "").strip()
        prompt = draft or (config.answer_prompt or "").strip() or ANSWER_PROMPT.strip()
        prompt_source = prompt_source_of(prompt, config)
        # the persona exactly as a call composes it, then the rules under test
        session = ChatSession(settings=eval_settings, config=config, knowledge=None)
        system = f"{session.answer_prompt.rstrip()}\n\n{prompt}\n"
        run_id = str(uuid.uuid4())
        await st.create_run(
            run_id,
            prompt=prompt,
            prompt_source=prompt_source,
            config_id=config.id,
            model=session.model,
            total=len(items),
            max_sheets=body.max_sheets,
            min_score=body.min_score,
            dataset=dataset,
        )
        running = True
        gate = asyncio.Semaphore(CONCURRENCY)

        async def one(item: dict) -> None:
            async with gate:
                outcome = await answer_item(
                    item,
                    session=session,
                    system=system,
                    max_sheets=body.max_sheets,
                    min_score=body.min_score,
                )
            await st.add_result(run_id, item, outcome)

        try:
            await asyncio.gather(*(one(i) for i in items))
            await st.finish_run(run_id, "done", None)
        except Exception as exc:
            logger.exception(f"[ANSWER-EVAL] run {run_id[:8]} failed")
            await st.finish_run(run_id, "failed", str(exc)[:500])
        finally:
            running = False
        logger.info(
            f"[ANSWER-EVAL] run {run_id[:8]}: {len(items)} items of {dataset!r}, "
            f"prompt {prompt_source}, model {session.model}, config #{config.id}"
        )
        return {"run": await st.run(run_id), "results": await st.results(run_id)}

    @router.get("/runs")
    async def api_runs(request: Request):
        return {"runs": await store(request).runs(), "running": running}

    @router.get("/runs/{run_id}")
    async def api_run_detail(request: Request, run_id: uuid.UUID):
        st = store(request)
        run = await st.run(str(run_id))
        if run is None:
            raise HTTPException(404, "run not found")
        return {"run": run, "results": await st.results(str(run_id))}

    @router.patch("/runs/{run_id}/results/{result_id}")
    async def api_judge(request: Request, run_id: uuid.UUID, result_id: int, body: JudgeIn):
        st = store(request)
        if not await st.judge(str(run_id), result_id, body.ok, (body.note or "").strip() or None):
            raise HTTPException(404, "result not found")
        return {"run": await st.run(str(run_id))}

    @router.delete("/runs/{run_id}")
    async def api_delete_run(request: Request, run_id: uuid.UUID):
        if not await store(request).delete_run(str(run_id)):
            raise HTTPException(404, "run not found")
        return {"deleted": True}

    return router
