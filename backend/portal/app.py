"""FastAPI backend for the portal: JSON API plus the built React app."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voice_agent.config import Settings
from voice_agent.knowledge.embeddings import OpenAIEmbedder
from voice_agent.knowledge.extract import SUPPORTED_EXTENSIONS, UnsupportedFileError
from voice_agent.knowledge.service import EmptyDocumentError, KnowledgeService
from voice_agent.knowledge.store import KnowledgeStore
from voice_agent.knowledge.translate import SarvamTranslator
from voice_agent.languages import SUPPORTED_LANGUAGES, UnsupportedLanguageError, get_language
from voice_agent.prompts import DEFAULT_PROMPT_TEMPLATE, build_system_prompt
from voice_agent.tracking.config_store import (
    AgentConfig,
    list_config_history,
    load_active_config,
    save_config,
)
from voice_agent.tracking.db import create_pool, ensure_schema

# repo/frontend/dist, built with `npm run build` in frontend/
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

CALLS_QUERY = """
select c.id, c.transport, c.direction, c.from_number, c.to_number, c.language,
       c.started_at, c.ended_at, c.end_reason, c.turn_count, c.config_id,
       extract(epoch from (coalesce(c.ended_at, now()) - c.started_at))::int as duration_secs,
       round(avg(t.response_latency_ms))::int as avg_latency_ms,
       max(t.response_latency_ms)::int as max_latency_ms,
       bool_or(t.interrupted) as had_interruption
from calls c left join turns t on t.call_id = c.id
group by c.id
order by c.started_at desc
limit $1
"""

TURNS_QUERY = """
select turn_index, user_text, assistant_text, user_spoke_at, bot_spoke_at, completed_at,
       response_latency_ms, turn_release_ms, stt_ttfb_ms, llm_ttfb_ms, tts_ttfb_ms, interrupted,
       contributions::text as contributions, searches::text as searches
from turns where call_id = $1 order by turn_index
"""

STATS_QUERY = """
select
  (select count(*) from calls) as calls_total,
  (select count(*) from calls where started_at >= date_trunc('day', now())) as calls_today,
  (select count(*) from calls where ended_at is null
      and started_at > now() - interval '2 hours') as live_calls,
  (select count(*) from turns) as turns_total,
  (select round(avg(response_latency_ms))::int from turns) as avg_latency_ms,
  (select percentile_cont(0.95) within group (order by response_latency_ms)::int
      from turns where response_latency_ms is not null) as p95_latency_ms,
  (select count(*) from turns where interrupted) as interrupted_turns
"""

BY_TRANSPORT_QUERY = """
select c.transport, count(distinct c.id)::int as calls,
       round(avg(t.response_latency_ms))::int as avg_latency_ms
from calls c left join turns t on t.call_id = c.id
group by c.transport order by calls desc
"""

LATENCY_BUCKETS_QUERY = """
select bucket, count(*)::int as turns from (
  select case
    when response_latency_ms < 1000 then 0
    when response_latency_ms < 1500 then 1
    when response_latency_ms < 2000 then 2
    when response_latency_ms < 3000 then 3
    else 4 end as bucket
  from turns where response_latency_ms is not null
) b group by bucket order by bucket
"""
LATENCY_BUCKET_LABELS = ["< 1 s", "1 – 1.5 s", "1.5 – 2 s", "2 – 3 s", "> 3 s"]


MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class SearchIn(BaseModel):
    question: str
    limit: int = 5
    document_id: str | None = None


class ConfigIn(BaseModel):
    system_prompt: str | None = None
    temperature: float | None = None
    language: str | None = None
    greeting: str | None = None
    note: str | None = None


def create_app(
    database_url: str,
    *,
    default_prompt_language: str,
    agent_name: str,
    knowledge: Settings | None = None,
) -> FastAPI:
    """Build the portal app bound to ``database_url``.

    ``knowledge`` supplies the Qdrant/embedding settings; without it the
    knowledge routes report themselves as unavailable.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = await create_pool(database_url)
        await ensure_schema(app.state.pool)
        app.state.knowledge = None
        app.state.knowledge_store = None
        if knowledge is not None:
            app.state.knowledge_store, app.state.knowledge = _build_knowledge(
                app.state.pool, knowledge
            )
        try:
            yield
        finally:
            if app.state.knowledge_store is not None:
                await app.state.knowledge_store.close()
            await app.state.pool.close()

    app = FastAPI(title="Voice agent portal", lifespan=lifespan)
    languages = [
        {"code": lang.code, "english_name": lang.english_name, "native_name": lang.native_name}
        for lang in SUPPORTED_LANGUAGES.values()
    ]

    def pool(request: Request) -> asyncpg.Pool:
        return request.app.state.pool

    @app.get("/api/stats")
    async def api_stats(request: Request):
        async with pool(request).acquire() as conn:
            row = await conn.fetchrow(STATS_QUERY)
            by_transport = await conn.fetch(BY_TRANSPORT_QUERY)
            buckets = {r["bucket"]: r["turns"] for r in await conn.fetch(LATENCY_BUCKETS_QUERY)}
        return {
            **dict(row),
            "by_transport": [dict(r) for r in by_transport],
            "latency_buckets": [
                {"label": label, "turns": buckets.get(i, 0)}
                for i, label in enumerate(LATENCY_BUCKET_LABELS)
            ],
        }

    @app.get("/api/calls")
    async def api_calls(request: Request, limit: int = 100):
        async with pool(request).acquire() as conn:
            return [dict(r) for r in await conn.fetch(CALLS_QUERY, min(limit, 1000))]

    @app.get("/api/calls/{call_id}")
    async def api_call(request: Request, call_id: uuid.UUID):
        async with pool(request).acquire() as conn:
            call = await conn.fetchrow("select * from calls where id = $1", call_id)
            if call is None:
                raise HTTPException(404, "call not found")
            turns = await conn.fetch(TURNS_QUERY, call_id)
        return {"call": dict(call), "turns": [_turn_json(t) for t in turns]}

    @app.get("/api/config")
    async def api_config(request: Request):
        config = await load_active_config(pool(request))
        return {
            "id": config.id,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "language": config.language,
            "greeting": config.greeting,
            "default_language": default_prompt_language,
            "default_prompt": DEFAULT_PROMPT_TEMPLATE,
            "agent_name": agent_name,
            "languages": languages,
        }

    @app.get("/api/config/preview")
    async def api_config_preview(language: str, system_prompt: str | None = None):
        """Render a prompt (or the built-in one) for a language, as the agent will see it."""
        try:
            lang = get_language(language)
            text = build_system_prompt(agent_name, lang, override=system_prompt or None)
        except (UnsupportedLanguageError, KeyError, IndexError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"language": lang.code, "prompt": text}

    @app.get("/api/config/history")
    async def api_config_history(request: Request):
        return [dict(r) for r in await list_config_history(pool(request))]

    @app.post("/api/config", status_code=201)
    async def api_save_config(request: Request, body: ConfigIn):
        try:
            config = parse_config(body.system_prompt, body.temperature, body.language)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        new_id = await save_config(
            pool(request),
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            language=config.language,
            greeting=(body.greeting or "").strip() or None,
            note=(body.note or "").strip() or None,
        )
        return {"id": new_id}

    # ---- knowledge base ---------------------------------------------------

    def knowledge_service(request: Request) -> KnowledgeService:
        service = request.app.state.knowledge
        if service is None:
            raise HTTPException(
                503,
                "Knowledge base unavailable: set OPENAI_API_KEY and restart the portal",
            )
        return service

    @app.get("/api/knowledge/status")
    async def api_knowledge_status(request: Request):
        store: KnowledgeStore | None = request.app.state.knowledge_store
        qdrant_ok = await store.ping() if store else False
        points = None
        if store and qdrant_ok:
            try:
                points = await store.point_count()
            except Exception:
                points = None
        return {
            "qdrant_ok": qdrant_ok,
            "qdrant_url": knowledge.qdrant_url if knowledge else None,
            "collection": knowledge.qdrant_collection if knowledge else None,
            "points": points,
            "embedding_model": knowledge.embedding_model if knowledge else None,
            "embedding_key_set": request.app.state.knowledge is not None,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        }

    @app.get("/api/documents")
    async def api_documents(request: Request):
        async with pool(request).acquire() as conn:
            rows = await conn.fetch("select * from documents order by created_at desc")
        return [dict(r) for r in rows]

    @app.post("/api/documents", status_code=201)
    async def api_upload_document(request: Request, file: UploadFile):
        service = knowledge_service(request)
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
        try:
            result = await service.ingest(file.filename or "upload", data, file.content_type)
        except (UnsupportedFileError, EmptyDocumentError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:  # embedding or Qdrant failure; the document row records it
            raise HTTPException(502, f"indexing failed: {exc}") from exc
        async with pool(request).acquire() as conn:
            row = await conn.fetchrow(
                "select * from documents where id = $1", uuid.UUID(result.document_id)
            )
        return dict(row)

    @app.delete("/api/documents/{document_id}")
    async def api_delete_document(request: Request, document_id: uuid.UUID):
        deleted = await knowledge_service(request).delete(str(document_id))
        if not deleted:
            raise HTTPException(404, "document not found")
        return {"deleted": True}

    @app.post("/api/knowledge/search")
    async def api_knowledge_search(request: Request, body: SearchIn):
        if not body.question.strip():
            raise HTTPException(422, "question is empty")
        result = await knowledge_service(request).search(
            body.question.strip(),
            limit=max(1, min(body.limit, 20)),
            document_id=body.document_id or None,
        )
        return {
            "hits": [h.__dict__ for h in result.hits],
            "embed_ms": result.embed_ms,
            "search_ms": result.search_ms,
            "translate_ms": result.translate_ms,
            "translated_query": result.translated_query,
        }

    _mount_frontend(app)
    return app


def _build_knowledge(
    pool: asyncpg.Pool, settings: Settings
) -> tuple[KnowledgeStore | None, KnowledgeService | None]:
    """Qdrant store is always built (for status); the service needs the embedding key."""
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    embedder = None
    if api_key:
        embedder = OpenAIEmbedder(
            api_key=api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            base_url=settings.embedding_base_url,
        )
    store = KnowledgeStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        collection=settings.qdrant_collection,
        vector_size=embedder.dimensions if embedder else 0,
    )
    if embedder is None:
        return store, None
    service = KnowledgeService(
        pool=pool,
        store=store,
        embedder=embedder,
        translator=SarvamTranslator(settings.sarvam_api_key.get_secret_value()),
        chunk_chars=settings.knowledge_chunk_chars,
        chunk_overlap=settings.knowledge_chunk_overlap,
    )
    return store, service


def _turn_json(row: asyncpg.Record) -> dict:
    turn = dict(row)
    for key in ("contributions", "searches"):
        raw = turn.get(key)
        turn[key] = json.loads(raw) if raw else []
    return turn


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built React app, with the SPA fallback for client-side routes."""
    if not FRONTEND_DIST.is_dir():

        @app.get("/{path:path}", include_in_schema=False)
        async def missing_build(path: str):
            return PlainTextResponse(
                "Frontend not built. Run: cd frontend && npm install && npm run build",
                status_code=503,
            )

        return

    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


def parse_config(
    system_prompt: str | None, temperature: float | None, language: str | None = None
) -> AgentConfig:
    """Validate a config submission. Empty prompt/language mean "use the built-in/env one"."""
    language = (language or "").strip() or None
    if language is not None:
        get_language(language)  # raises UnsupportedLanguageError (a ValueError) with the list
    prompt = (system_prompt or "").strip() or None
    if prompt is not None:
        try:
            build_system_prompt("Agent", get_language("hi-IN"), override=prompt)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"prompt has an invalid placeholder {exc}; allowed: "
                "{agent_name}, {language_english_name}, {language_native_name}"
            ) from exc
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    return AgentConfig(system_prompt=prompt, temperature=temperature, language=language)
