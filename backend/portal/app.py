"""FastAPI backend for the portal: JSON API plus the built React app."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voice_agent import models as catalogue
from voice_agent.config import Settings
from voice_agent.knowledge.embeddings import OpenAIEmbedder
from voice_agent.knowledge.extract import SUPPORTED_EXTENSIONS, UnsupportedFileError
from voice_agent.knowledge.rewrite import DEFAULT_REWRITE_PROMPT
from voice_agent.knowledge.service import EmptyDocumentError, KnowledgeService
from voice_agent.knowledge.store import KnowledgeStore
from voice_agent.knowledge.tool import PROMPT_ADDITION
from voice_agent.knowledge.translate import SarvamTranslator
from voice_agent.languages import SUPPORTED_LANGUAGES, UnsupportedLanguageError, get_language
from voice_agent.phrases import IDLE_CHECK_LINES, IDLE_GOODBYE_LINES
from voice_agent.prompts import DEFAULT_PROMPT_TEMPLATE, build_system_prompt
from voice_agent.pronunciation import SarvamDictionaries, entries_to_text, parse_entries
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
       contributions::text as contributions, searches::text as searches,
       prompt_tokens, completion_tokens, tts_characters, stt_audio_seconds
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
    history: list[str] | None = None
    """Query rewrite: the caller's earlier messages, oldest first."""


class ConfigIn(BaseModel):
    system_prompt: str | None = None
    temperature: float | None = None
    language: str | None = None
    greeting: str | None = None
    idle_prompt_secs: float | None = None
    idle_check_line: str | None = None
    idle_goodbye_line: str | None = None
    knowledge_prompt: str | None = None
    rewrite_prompt: str | None = None
    """None keeps the active version's rewrite prompt; "" resets it to the built-in text."""
    pronunciations: str | None = None
    """``word = spoken form`` lines; None keeps the active version's dictionary."""
    stt_model: str | None = None
    llm_model: str | None = None
    tts_model: str | None = None
    tts_speaker: str | None = None
    translate_model: str | None = None
    embedding_model: str | None = None
    """Per-step models; None or "" = environment default. A changed embedding model
    re-embeds every document before the version is saved."""
    note: str | None = None


class PronunciationTestIn(BaseModel):
    text: str
    pronunciations: str | None = None


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
            # the active config may pin a different embedding model than .env
            active = await load_active_config(app.state.pool)
            app.state.knowledge_store, app.state.knowledge = _build_knowledge(
                app.state.pool, _with_embedding(knowledge, active.embedding_model)
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
            "idle_prompt_secs": config.idle_prompt_secs,
            "idle_check_line": config.idle_check_line,
            "idle_goodbye_line": config.idle_goodbye_line,
            "knowledge_prompt": config.knowledge_prompt,
            "pronunciations": entries_to_text(config.pronunciations),
            "tts_dict_id": config.tts_dict_id,
            "stt_model": config.stt_model,
            "llm_model": config.llm_model,
            "tts_model": config.tts_model,
            "tts_speaker": config.tts_speaker,
            "translate_model": config.translate_model,
            "embedding_model": config.embedding_model,
            "model_defaults": {
                "stt": knowledge.sarvam_stt_model if knowledge else "saaras:v4",
                "llm": knowledge.llm_model if knowledge else "sarvam-105b",
                "tts": knowledge.sarvam_tts_model if knowledge else "bulbul:v3",
                "tts_speaker": (knowledge.sarvam_tts_speaker if knowledge else None) or "shubh",
                "translate": knowledge.translate_model if knowledge else "mayura:v1",
                "embedding": knowledge.embedding_model if knowledge else "text-embedding-3-small",
            },
            "knowledge_prompt_default": PROMPT_ADDITION.strip(),
            "rewrite_prompt": config.rewrite_prompt,
            "rewrite_prompt_default": DEFAULT_REWRITE_PROMPT,
            "query_rewrite_on": bool(knowledge and knowledge.knowledge_query_rewrite),
            "knowledge_on": bool(
                knowledge and knowledge.knowledge_enabled and knowledge.openai_api_key
            ),
            "idle_defaults": {
                "prompt_secs": knowledge.idle_prompt_secs if knowledge else 10,
                "check_lines": IDLE_CHECK_LINES,
                "goodbye_lines": IDLE_GOODBYE_LINES,
            },
            "default_language": default_prompt_language,
            "default_prompt": DEFAULT_PROMPT_TEMPLATE,
            "agent_name": agent_name,
            "languages": languages,
        }

    @app.get("/api/models")
    async def api_models(language: str | None = None):
        """Sarvam models per pipeline step that support ``language`` (default: the env language)."""
        try:
            lang = get_language(language or default_prompt_language)
        except UnsupportedLanguageError as exc:
            raise HTTPException(422, str(exc)) from exc
        return catalogue.catalogue(lang.code)

    @app.get("/api/config/preview")
    async def api_config_preview(
        language: str, system_prompt: str | None = None, knowledge_prompt: str | None = None
    ):
        """Render a prompt (or the built-in one) for a language, as the agent will see it."""
        try:
            lang = get_language(language)
            text = build_system_prompt(agent_name, lang, override=system_prompt or None)
        except (UnsupportedLanguageError, KeyError, IndexError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if knowledge and knowledge.knowledge_enabled and knowledge.openai_api_key:
            text += "\n\n" + (knowledge_prompt or PROMPT_ADDITION).strip() + "\n"
        return {"language": lang.code, "prompt": text}

    @app.get("/api/config/history")
    async def api_config_history(request: Request):
        return await list_config_history(pool(request))

    @app.post("/api/config", status_code=201)
    async def api_save_config(request: Request, body: ConfigIn):
        try:
            config = parse_config(body.system_prompt, body.temperature, body.language)
            if body.idle_prompt_secs is not None and not 0 <= body.idle_prompt_secs <= 120:
                raise ValueError("idle_prompt_secs must be between 0 and 120 seconds")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        active = await load_active_config(pool(request))
        language = config.language or default_prompt_language
        models = {
            key: (getattr(body, key) or "").strip() or None
            for key in (
                "stt_model",
                "llm_model",
                "tts_model",
                "tts_speaker",
                "translate_model",
                "embedding_model",
            )
        }
        try:
            catalogue.validate_selection(
                language,
                stt_model=models["stt_model"],
                llm_model=models["llm_model"],
                tts_model=models["tts_model"]
                or (knowledge.sarvam_tts_model if knowledge else None),
                tts_speaker=models["tts_speaker"],
                translate_model=models["translate_model"],
                embedding_model=models["embedding_model"],
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        reindex = None
        if knowledge is not None and request.app.state.knowledge is not None:
            wanted = models["embedding_model"] or knowledge.embedding_model
            if wanted != request.app.state.knowledge.embedding_model:
                # Rebuild the collection for the new model before the version exists,
                # so a failed re-index never leaves a config the agent cannot search with.
                store, service = _build_knowledge(pool(request), _with_embedding(knowledge, wanted))
                assert service is not None
                try:
                    reindex = await service.reindex_all()
                except Exception as exc:
                    await store.close()
                    raise HTTPException(502, f"Re-indexing with {wanted} failed: {exc}") from exc
                old_store = request.app.state.knowledge_store
                request.app.state.knowledge_store, request.app.state.knowledge = store, service
                if old_store is not None:
                    # requests already running on the old store (an upload, a search)
                    # must finish first; close it once they are long done
                    asyncio.get_running_loop().create_task(_close_later(old_store, 60))
        pronunciations = active.pronunciations
        dict_id = active.tts_dict_id
        if body.pronunciations is not None:
            try:
                entries = parse_entries(body.pronunciations)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if knowledge is None:
                raise HTTPException(503, "pronunciation dictionaries need the Sarvam key")
            dictionaries = SarvamDictionaries(knowledge.sarvam_api_key.get_secret_value())
            try:
                dict_id = await dictionaries.replace(language, entries, active.tts_dict_id)
            except Exception as exc:
                raise HTTPException(502, f"Sarvam dictionary update failed: {exc}") from exc
            pronunciations = [{"word": e.word, "spoken": e.spoken} for e in entries]
        new_id = await save_config(
            pool(request),
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            language=config.language,
            greeting=(body.greeting or "").strip() or None,
            idle_prompt_secs=body.idle_prompt_secs,
            idle_check_line=(body.idle_check_line or "").strip() or None,
            idle_goodbye_line=(body.idle_goodbye_line or "").strip() or None,
            knowledge_prompt=(body.knowledge_prompt or "").strip() or None,
            rewrite_prompt=(
                active.rewrite_prompt
                if body.rewrite_prompt is None
                else body.rewrite_prompt.strip() or None
            ),
            pronunciations=pronunciations,
            tts_dict_id=dict_id,
            **models,
            note=(body.note or "").strip() or None,
        )
        return {
            "id": new_id,
            "tts_dict_id": dict_id,
            "reindexed": reindex.reindexed if reindex else None,
            "needs_reupload": reindex.needs_reupload if reindex else None,
        }

    @app.post("/api/pronunciation/test")
    async def api_pronunciation_test(request: Request, body: PronunciationTestIn):
        """Synthesise ``text`` with the active dictionary (or a temporary one from the
        submitted entries) and return WAV audio as base64, for listening in the browser."""
        if knowledge is None:
            raise HTTPException(503, "needs the Sarvam key")
        active = await load_active_config(pool(request))
        language = active.language or default_prompt_language
        api_key = knowledge.sarvam_api_key.get_secret_value()
        dict_id, temporary = active.tts_dict_id, False
        if body.pronunciations is not None:
            try:
                entries = parse_entries(body.pronunciations)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if entries:
                dict_id = await SarvamDictionaries(api_key).create(language, entries)
                temporary = True
            else:
                dict_id = None
        try:
            audio = await _synthesise(
                api_key,
                body.text,
                language,
                knowledge,
                dict_id,
                model=active.tts_model,
                speaker=active.tts_speaker,
            )
        finally:
            if temporary and dict_id:
                await SarvamDictionaries(api_key).delete(dict_id)
        return {"audio_base64": audio, "dict_id": dict_id}

    # ---- knowledge base ---------------------------------------------------

    def knowledge_service(request: Request) -> KnowledgeService:
        service = request.app.state.knowledge
        if service is None:
            raise HTTPException(
                503,
                "Knowledge base unavailable: set OPENAI_API_KEY and restart the portal",
            )
        return service

    @app.get("/api/test-call")
    async def api_test_call():
        """What the browser needs to place a WebRTC test call: the bot URL and whether
        the bot answers and has the WebRTC transport enabled (``BOT_TRANSPORT`` unset or
        ``webrtc``)."""
        import httpx

        if knowledge is None:
            return {"bot_url": None, "reachable": False, "webrtc": None}
        url = knowledge.bot_public_url.rstrip("/")
        reachable, webrtc = False, None
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{url}/status")
                reachable = r.status_code == 200
        except Exception:
            reachable = False
        if reachable:
            webrtc = knowledge.bot_transport in (None, "webrtc")
        return {"bot_url": url, "reachable": reachable, "webrtc": webrtc}

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
            "embedding_model": (
                request.app.state.knowledge.embedding_model
                if request.app.state.knowledge
                else knowledge.embedding_model
                if knowledge
                else None
            ),
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
        service = knowledge_service(request)
        limit = max(1, min(body.limit, 20))
        result = None
        if service.query_rewrite:  # same as on calls: caller's messages -> rewrite -> search
            caller = [str(m) for m in (body.history or [])] + [body.question.strip()]
            config = await load_active_config(pool(request))
            result = await service.search_rewritten(
                caller[-10:],
                prompt=config.rewrite_prompt,
                limit=limit,
                document_id=body.document_id or None,
            )
        if result is None:
            result = await service.search(
                body.question.strip(), limit=limit, document_id=body.document_id or None
            )
        return {
            "hits": [h.__dict__ for h in result.hits],
            "embed_ms": result.embed_ms,
            "search_ms": result.search_ms,
            "translate_ms": result.translate_ms,
            "translated_query": result.translated_query,
            "rewritten_query": result.rewritten_query,
            "rewrite_ms": result.rewrite_ms,
        }

    _mount_frontend(app)
    return app


async def _synthesise(
    api_key: str,
    text: str,
    language: str,
    settings: Settings,
    dict_id: str | None,
    *,
    model: str | None = None,
    speaker: str | None = None,
) -> str:
    """One REST synthesis with the agent's voice (portal choices first, then .env)."""
    import base64

    import httpx

    payload = {
        "text": text,
        "target_language_code": language,
        "speaker": speaker
        or settings.sarvam_tts_speaker
        or catalogue.default_speaker(model or settings.sarvam_tts_model, language)
        or "shubh",
        "model": model or settings.sarvam_tts_model,
        "speech_sample_rate": 24000,
        "output_audio_codec": "wav",
        "pace": settings.sarvam_tts_pace,
    }
    if dict_id:
        payload["dict_id"] = dict_id
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": api_key},
            json=payload,
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Sarvam TTS failed: {r.text[:200]}")
    audio = r.json()["audios"][0]
    base64.b64decode(audio)  # validate
    return audio


async def _close_later(store: KnowledgeStore, delay_secs: float) -> None:
    await asyncio.sleep(delay_secs)
    with contextlib.suppress(Exception):  # best effort: the process may be shutting down
        await store.close()


def _with_embedding(settings: Settings, model: str | None) -> Settings:
    """Settings with the portal-chosen embedding model (dimensions come from the model)."""
    if not model or model == settings.embedding_model:
        return settings
    return settings.model_copy(update={"embedding_model": model, "embedding_dimensions": None})


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
    from voice_agent.services import build_query_rewriter

    service = KnowledgeService(
        pool=pool,
        store=store,
        embedder=embedder,
        rewriter=build_query_rewriter(settings),
        translator=SarvamTranslator(
            settings.sarvam_api_key.get_secret_value(), source_language=settings.agent_language
        ),
        chunk_chars=settings.knowledge_chunk_chars,
        chunk_overlap=settings.knowledge_chunk_overlap,
        whole_document=settings.knowledge_whole_document,
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
