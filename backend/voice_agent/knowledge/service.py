"""Ingestion and retrieval over the knowledge store, with document records in Postgres."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import asyncpg
from loguru import logger

from voice_agent.knowledge.chunk import chunk_text
from voice_agent.knowledge.embeddings import Embedder
from voice_agent.knowledge.extract import extract_text
from voice_agent.knowledge.store import Hit, KnowledgeStore
from voice_agent.knowledge.translate import Translator, needs_translation


class EmptyDocumentError(ValueError):
    """The file contained no extractable text."""


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    name: str
    chars: int
    chunk_count: int
    seconds: float


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    embed_ms: int
    search_ms: int
    translated_query: str | None = None
    """English form of the question actually embedded, when it was translated."""
    translate_ms: int = 0


class KnowledgeService:
    def __init__(
        self,
        *,
        pool: asyncpg.Pool | None,
        store: KnowledgeStore,
        embedder: Embedder,
        translator: Translator | None = None,
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
        whole_document: bool = False,
    ) -> None:
        self._pool = pool
        self._store = store
        self._embedder = embedder
        self._translator = translator
        self._chunk_chars = chunk_chars
        self._chunk_overlap = chunk_overlap
        self._whole_document = whole_document

    @property
    def qdrant_url(self) -> str:
        return self._store.url

    async def ping(self) -> bool:
        """True when Qdrant answers; used for start-up warnings."""
        return await self._store.ping()

    async def ingest(self, filename: str, data: bytes, media_type: str | None) -> IngestResult:
        """Extract, chunk, embed and store one file; records progress in ``documents``."""
        if self._pool is None:
            raise RuntimeError("ingest needs the documents database")
        started = time.monotonic()
        text = extract_text(filename, data)
        chunks = chunk_text(
            text,
            max_chars=self._chunk_chars,
            overlap=self._chunk_overlap,
            whole_document=self._whole_document,
        )
        if not chunks:
            raise EmptyDocumentError(f"No text could be extracted from {filename!r}")
        document_id = str(uuid.uuid4())
        await self._record(
            document_id, filename, media_type, len(data), len(text), len(chunks), "embedding"
        )
        try:
            await self._store.ensure_collection()
            vectors = await self._embedder.embed([c.text for c in chunks])
            await self._store.upsert(document_id, filename, chunks, vectors)
        except Exception as exc:
            await self._set_status(document_id, "failed", str(exc)[:500])
            raise
        await self._set_status(document_id, "ready", None)
        seconds = time.monotonic() - started
        logger.info(
            f"Ingested {filename!r}: {len(chunks)} chunks, {len(text)} chars in {seconds:.1f}s"
        )
        return IngestResult(document_id, filename, len(text), len(chunks), seconds)

    async def search(
        self, question: str, *, limit: int = 5, document_id: str | None = None
    ) -> SearchResult:
        """Embed the question, translated to English first when it is in an Indic script."""
        translated: str | None = None
        translate_ms = 0
        if self._translator is not None and needs_translation(question):
            logger.info(f"Knowledge: translating {question!r} to English for retrieval")
            t = time.monotonic()
            translated = await self._translator.to_english(question)
            translate_ms = round((time.monotonic() - t) * 1000)
        t0 = time.monotonic()
        [vector] = await self._embedder.embed([translated or question])
        t1 = time.monotonic()
        hits = await self._store.search(vector, limit=limit, document_id=document_id)
        t2 = time.monotonic()
        return SearchResult(
            hits, round((t1 - t0) * 1000), round((t2 - t1) * 1000), translated, translate_ms
        )

    async def delete(self, document_id: str) -> bool:
        if self._pool is None:
            raise RuntimeError("delete needs the documents database")
        async with self._pool.acquire() as conn:
            deleted = await conn.fetchval(
                "delete from documents where id = $1 returning id", uuid.UUID(document_id)
            )
        if deleted is None:
            return False
        await self._store.delete_document(document_id)
        return True

    async def list_documents(self) -> list[asyncpg.Record]:
        if self._pool is None:
            return []
        async with self._pool.acquire() as conn:
            return await conn.fetch("select * from documents order by created_at desc")

    async def _record(  # pool is checked by ingest()
        self,
        document_id: str,
        name: str,
        media_type: str | None,
        size: int,
        chars: int,
        n: int,
        status: str,
    ) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "insert into documents (id, name, media_type, size_bytes, chars, chunk_count, "
                "embedding_model, collection, status, created_at) "
                "values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
                uuid.UUID(document_id),
                name,
                media_type,
                size,
                chars,
                n,
                self._embedder.model,
                self._store.collection,
                status,
                datetime.now(UTC),
            )

    async def _set_status(self, document_id: str, status: str, error: str | None) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "update documents set status = $2, error = $3 where id = $1",
                uuid.UUID(document_id),
                status,
                error,
            )
