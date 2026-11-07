"""Rewrite-prompt evaluation: a fixed set of caller questions run through one rewrite prompt.

The query rewrite turns a caller's Tamil words into one English search question. To
improve its prompt you need the same questions every time: a small fixed set (seeded
with random questions from the evaluation bank, then edited by hand) that is rewritten
with a draft prompt and searched, so the rewritten queries and the rank of the expected
sheet can be compared run after run. Nothing here changes the saved configuration.

Mounted by the portal under ``/api/rewrite-evals``.
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

from portal.evals import _json, eval_settings_for, source_matches
from voice_agent.config import Settings
from voice_agent.knowledge.rewrite import DEFAULT_REWRITE_PROMPT, QueryRewriter
from voice_agent.knowledge.service import KnowledgeService, SearchType
from voice_agent.knowledge.tool import KnowledgeTool
from voice_agent.tracking.config_store import load_active_config

SEED_COUNT = 20
CONCURRENCY = 2
REWRITE_TIMEOUT_SECS = 15.0
"""Longer than a call's rewrite timeout: here we judge the prompt's output, not its speed.
The page flags rewrites slower than the call limit."""


@dataclass(frozen=True)
class RewriteOutcome:
    rewritten_query: str | None
    hits: list[dict]
    expected_rank: int | None
    expected_sent: bool
    rewrite_ms: int
    ms: int
    error: str | None


async def evaluate_question(
    question_ta: str,
    expected_source: str,
    *,
    rewriter: Any,
    prompt: str,
    service: Any,
    limit: int,
    min_score: float,
) -> RewriteOutcome:
    """Rewrite one question with ``prompt`` and search with the result, as a call does."""
    started = time.monotonic()
    rewrite = await rewriter.rewrite([question_ta], prompt)
    if rewrite is None:  # one retry: a Sarvam hiccup should not count against the prompt
        rewrite = await rewriter.rewrite([question_ta], prompt)
    rewrite_ms = round((time.monotonic() - started) * 1000)
    if rewrite is None:
        return RewriteOutcome(
            None, [], None, False, rewrite_ms, rewrite_ms, "rewrite failed or timed out"
        )
    try:
        result = await service.search(rewrite.query, limit=limit)
    except Exception as exc:
        ms = round((time.monotonic() - started) * 1000)
        return RewriteOutcome(rewrite.query, [], None, False, rewrite_ms, ms, str(exc)[:300])
    selector = KnowledgeTool(service=service, limit=limit, min_score=min_score)
    chosen = {id(h) for h in selector._selected(result.hits)}
    hits = [
        {
            "source": h.document_name,
            "document_id": h.document_id,
            "chunk_index": h.chunk_index,
            "score": round(h.score, 3),
            "used": id(h) in chosen,
        }
        for h in result.hits
    ]
    rank = next(
        (i for i, h in enumerate(hits, start=1) if source_matches(h["source"], expected_source)),
        None,
    )
    sent = bool(rank and hits[rank - 1]["used"])
    ms = round((time.monotonic() - started) * 1000)
    return RewriteOutcome(rewrite.query, hits, rank, sent, rewrite_ms, ms, None)


# ---------- storage ----------

QUESTION_COLUMNS = (
    "id, question_ta, question_en, expected_source, source_question_id, created_at, updated_at"
)
RUN_COLUMNS = (
    "r.id, r.started_at, r.finished_at, r.status, r.prompt, r.prompt_source, r.config_id, "
    "r.model, r.collection, r.search_type, r.total, r.error"
)
RUN_SUMMARY_SQL = """
    count(x.id)                                     as results,
    count(x.expected_rank)                          as retrieved,
    count(x.id) filter (where x.expected_sent)      as sent,
    count(x.id) filter (where x.expected_rank = 1)  as rank1,
    count(x.error)                                  as errors,
    count(x.id) filter (where x.judged_ok)          as judged_ok,
    coalesce(round(avg(x.rewrite_ms))::int, 0)      as avg_rewrite_ms
"""


class PostgresRewriteStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def questions(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"select {QUESTION_COLUMNS} from rewrite_eval_questions order by id"
            )
        return [_json(r) for r in rows]  # type: ignore[misc]

    async def add_random_from_bank(self, count: int) -> int:
        """Copy ``count`` random enabled bank questions (with Tamil) not already in the set."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "insert into rewrite_eval_questions "
                "(question_ta, question_en, expected_source, source_question_id) "
                "select q.question_ta, q.question_en, q.expected_source, q.id "
                "from eval_questions q "
                "where q.enabled and q.question_ta is not null and q.question_ta <> '' "
                "and q.id not in (select source_question_id from rewrite_eval_questions "
                "where source_question_id is not null) order by random() limit $1",
                count,
            )
        return int(result.rsplit(" ", 1)[-1])

    async def add_question(
        self, question_ta: str, question_en: str | None, expected_source: str
    ) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "insert into rewrite_eval_questions (question_ta, question_en, expected_source) "
                f"values ($1, $2, $3) returning {QUESTION_COLUMNS}",
                question_ta,
                question_en,
                expected_source,
            )
        return _json(row)  # type: ignore[return-value]

    async def update_question(
        self, question_id: int, question_ta: str, question_en: str | None, expected_source: str
    ) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "update rewrite_eval_questions set question_ta = $2, question_en = $3, "
                "expected_source = $4, updated_at = now() where id = $1 "
                f"returning {QUESTION_COLUMNS}",
                question_id,
                question_ta,
                question_en,
                expected_source,
            )
        return _json(row)

    async def delete_question(self, question_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "delete from rewrite_eval_questions where id = $1", question_id
            )
        return result.endswith(" 1")

    async def create_run(
        self,
        run_id: str,
        *,
        prompt: str,
        prompt_source: str,
        config_id: int | None,
        model: str,
        collection: str,
        search_type: str,
        total: int,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "insert into rewrite_eval_runs (id, status, prompt, prompt_source, config_id, "
                "model, collection, search_type, total) "
                "values ($1, 'running', $2, $3, $4, $5, $6, $7, $8)",
                uuid.UUID(run_id),
                prompt,
                prompt_source,
                config_id,
                model,
                collection,
                search_type,
                total,
            )

    async def add_result(self, run_id: str, question: dict, outcome: RewriteOutcome) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "insert into rewrite_eval_results (run_id, question_id, question_ta, "
                "expected_source, rewritten_query, hits, expected_rank, expected_sent, "
                "rewrite_ms, ms, error) "
                "values ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)",
                uuid.UUID(run_id),
                question["id"],
                question["question_ta"],
                question["expected_source"],
                outcome.rewritten_query,
                json.dumps(outcome.hits, ensure_ascii=False),
                outcome.expected_rank,
                outcome.expected_sent,
                outcome.rewrite_ms,
                outcome.ms,
                outcome.error,
            )

    async def finish_run(self, run_id: str, status: str, error: str | None) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "update rewrite_eval_runs set status = $2, error = $3, finished_at = now() "
                "where id = $1",
                uuid.UUID(run_id),
                status,
                error,
            )

    async def runs(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"select {RUN_COLUMNS}, {RUN_SUMMARY_SQL} from rewrite_eval_runs r "
                "left join rewrite_eval_results x on x.run_id = r.id "
                "group by r.id order by r.started_at desc limit 100"
            )
        return [_json(r) for r in rows]  # type: ignore[misc]

    async def run(self, run_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"select {RUN_COLUMNS}, {RUN_SUMMARY_SQL} from rewrite_eval_runs r "
                "left join rewrite_eval_results x on x.run_id = r.id where r.id = $1 group by r.id",
                uuid.UUID(run_id),
            )
        return _json(row)

    async def results(self, run_id: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "select id, run_id, question_id, question_ta, expected_source, rewritten_query, "
                "hits, expected_rank, expected_sent, rewrite_ms, ms, error, at, judged_ok "
                "from rewrite_eval_results where run_id = $1 order by question_id",
                uuid.UUID(run_id),
            )
        return [_json(r) for r in rows]  # type: ignore[misc]

    async def judge(self, run_id: str, result_id: int, ok: bool) -> bool:
        """Record the reviewer's verdict on one rewritten query."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "update rewrite_eval_results set judged_ok = $3 where id = $2 and run_id = $1",
                uuid.UUID(run_id),
                result_id,
                ok,
            )
        return result.endswith(" 1")

    async def delete_run(self, run_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "delete from rewrite_eval_runs where id = $1", uuid.UUID(run_id)
            )
        return result.endswith(" 1")


# ---------- routes ----------


class QuestionIn(BaseModel):
    question_ta: str
    question_en: str | None = None
    expected_source: str


class RandomIn(BaseModel):
    count: int = 5


class JudgeIn(BaseModel):
    ok: bool
    """True when the reviewer finds the rewritten query correct for the caller's words."""


class RunIn(BaseModel):
    prompt: str | None = None
    """The rewrite prompt to try; empty = the active configuration's (or the built-in)."""
    search_type: SearchType = "semantic"


def create_rewrite_evals_router(settings: Settings | None) -> APIRouter:
    router = APIRouter(prefix="/api/rewrite-evals")
    running = False

    def store(request: Request) -> PostgresRewriteStore:
        return PostgresRewriteStore(request.app.state.pool)

    def agent_settings() -> Settings:
        if settings is None:
            raise HTTPException(503, "evaluation needs the Sarvam key in backend/.env")
        return settings

    @router.get("/questions")
    async def api_questions(request: Request):
        st = store(request)
        rows = await st.questions()
        seeded = 0
        if not rows:
            seeded = await st.add_random_from_bank(SEED_COUNT)
            rows = await st.questions()
        return {"questions": rows, "seeded": seeded}

    @router.post("/questions", status_code=201)
    async def api_add_question(request: Request, body: QuestionIn):
        ta, src = body.question_ta.strip(), body.expected_source.strip()
        if not ta or not src:
            raise HTTPException(422, "Tamil question and expected sheet are required")
        en = (body.question_en or "").strip() or None
        return await store(request).add_question(ta, en, src)

    @router.post("/questions/random", status_code=201)
    async def api_add_random(request: Request, body: RandomIn):
        count = max(1, min(body.count, 50))
        added = await store(request).add_random_from_bank(count)
        return {"added": added}

    @router.put("/questions/{question_id}")
    async def api_update_question(request: Request, question_id: int, body: QuestionIn):
        ta, src = body.question_ta.strip(), body.expected_source.strip()
        if not ta or not src:
            raise HTTPException(422, "Tamil question and expected sheet are required")
        en = (body.question_en or "").strip() or None
        row = await store(request).update_question(question_id, ta, en, src)
        if row is None:
            raise HTTPException(404, "question not found")
        return row

    @router.delete("/questions/{question_id}")
    async def api_delete_question(request: Request, question_id: int):
        if not await store(request).delete_question(question_id):
            raise HTTPException(404, "question not found")
        return {"deleted": True}

    @router.post("/runs", status_code=201)
    async def api_run(request: Request, body: RunIn):
        nonlocal running
        if running:
            raise HTTPException(409, "a rewrite run is already in progress")
        s = agent_settings()
        if request.app.state.knowledge is None:
            raise HTTPException(503, "knowledge search is not available (OPENAI_API_KEY, Qdrant)")
        st = store(request)
        questions = await st.questions()
        if not questions:
            raise HTTPException(422, "no questions to run")
        config = await load_active_config(request.app.state.pool)
        eval_settings = eval_settings_for(s, config)
        from voice_agent.services import build_knowledge_service

        service: KnowledgeService | None = build_knowledge_service(
            eval_settings, search_type=body.search_type
        )
        if service is None:
            raise HTTPException(503, "knowledge search is not available (OPENAI_API_KEY, Qdrant)")
        draft = (body.prompt or "").strip()
        configured = (config.rewrite_prompt or "").strip()
        prompt = draft or configured or DEFAULT_REWRITE_PROMPT
        prompt_source = (
            "default"
            if prompt.strip() == DEFAULT_REWRITE_PROMPT.strip()
            else "config"
            if prompt == configured
            else "draft"
        )
        rewriter = QueryRewriter(
            eval_settings.sarvam_api_key.get_secret_value(),
            model=eval_settings.llm_model,
            timeout=REWRITE_TIMEOUT_SECS,
        )
        run_id = str(uuid.uuid4())
        await st.create_run(
            run_id,
            prompt=prompt,
            prompt_source=prompt_source,
            config_id=config.id,
            model=eval_settings.llm_model,
            collection=eval_settings.qdrant_collection,
            search_type=body.search_type,
            total=len(questions),
        )
        running = True
        gate = asyncio.Semaphore(CONCURRENCY)

        async def one(question: dict) -> None:
            async with gate:
                outcome = await evaluate_question(
                    question["question_ta"],
                    question["expected_source"],
                    rewriter=rewriter,
                    prompt=prompt,
                    service=service,
                    limit=eval_settings.knowledge_top_k,
                    min_score=eval_settings.knowledge_min_score,
                )
            await st.add_result(run_id, question, outcome)

        try:
            await asyncio.gather(*(one(q) for q in questions))
            await st.finish_run(run_id, "done", None)
        except Exception as exc:
            logger.exception(f"[REWRITE-EVAL] run {run_id[:8]} failed")
            await st.finish_run(run_id, "failed", str(exc)[:500])
        finally:
            running = False
            await service.close()
        logger.info(
            f"[REWRITE-EVAL] run {run_id[:8]}: {len(questions)} questions, prompt {prompt_source}, "
            f"{body.search_type} search, config #{config.id}"
        )
        return {"run": await st.run(run_id), "results": await st.results(run_id)}

    @router.get("/runs")
    async def api_runs(request: Request):
        s = settings
        return {
            "runs": await store(request).runs(),
            "running": running,
            "call_timeout_ms": round(1000 * (s.knowledge_rewrite_timeout_secs if s else 3.0)),
        }

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
        if not await st.judge(str(run_id), result_id, body.ok):
            raise HTTPException(404, "result not found")
        return {"run": await st.run(str(run_id))}

    @router.delete("/runs/{run_id}")
    async def api_delete_run(request: Request, run_id: uuid.UUID):
        if not await store(request).delete_run(str(run_id)):
            raise HTTPException(404, "run not found")
        return {"deleted": True}

    return router
