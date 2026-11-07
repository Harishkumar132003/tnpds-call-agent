"""Qdrant storage for document chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from voice_agent.knowledge.chunk import Chunk
from voice_agent.knowledge.sparse import encode_document, encode_query

DENSE = "dense"
SPARSE = "bm25"
HYBRID_PREFETCH = 20  # candidates taken from each search before the RRF merge


class CollectionLayoutError(RuntimeError):
    """The collection predates the dense + sparse layout."""


@dataclass(frozen=True)
class Hit:
    score: float
    text: str
    document_id: str
    document_name: str
    chunk_index: int


class KnowledgeStore:
    """One Qdrant collection holding every document's chunks, filterable by document.

    Every point carries two vectors: ``dense`` (the embedding, cosine) and ``bm25``
    (a sparse term vector, IDF applied by Qdrant), so the same collection serves
    semantic, keyword and hybrid (RRF) search.
    """

    def __init__(self, *, url: str, api_key: str | None, collection: str, vector_size: int) -> None:
        self.url = url
        self.collection = collection
        self.vector_size = vector_size
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=30)

    async def ensure_collection(self) -> None:
        if await self._client.collection_exists(self.collection):
            await self.check_layout()
            return
        await self._client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        await self._client.create_payload_index(
            self.collection, "document_id", models.PayloadSchemaType.KEYWORD
        )

    async def check_layout(self) -> None:
        """Raise :class:`CollectionLayoutError` when the collection lacks the named
        ``dense`` and ``bm25`` vectors (made before keyword search existed)."""
        info = await self._client.get_collection(self.collection)
        params = info.config.params
        dense = params.vectors if isinstance(params.vectors, dict) else {}
        sparse = params.sparse_vectors or {}
        if DENSE not in dense or SPARSE not in sparse:
            raise CollectionLayoutError(
                f"collection {self.collection!r} has the old single-vector layout; "
                "reload it with scripts/kb_from_workbook.py (or re-index) to add BM25"
            )

    async def recreate_collection(self) -> None:
        """Drop the collection and create it again for ``vector_size``
        (after an embedding model change)."""
        if await self._client.collection_exists(self.collection):
            await self._client.delete_collection(self.collection)
        await self.ensure_collection()

    async def upsert(
        self, document_id: str, document_name: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.UUID(document_id), str(chunk.index))),
                vector={DENSE: vector, SPARSE: encode_document(chunk.text)},
                payload={
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "chars": len(chunk.text),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 128):
            await self._client.upsert(
                self.collection, points=points[start : start + 128], wait=True
            )

    async def search(
        self, vector: list[float], *, limit: int = 5, document_id: str | None = None
    ) -> list[Hit]:
        """Semantic: nearest embeddings, scored by cosine similarity."""
        response = await self._client.query_points(
            self.collection,
            query=vector,
            using=DENSE,
            limit=limit,
            query_filter=_filter(document_id),
            with_payload=True,
        )
        return _hits(response)

    async def search_keyword(
        self, text: str, *, limit: int = 5, document_id: str | None = None
    ) -> list[Hit]:
        """Keyword: BM25 over the chunk words (the score is unbounded, 0 = no word in common)."""
        sparse = encode_query(text)
        if not sparse.indices:
            return []
        response = await self._client.query_points(
            self.collection,
            query=sparse,
            using=SPARSE,
            limit=limit,
            query_filter=_filter(document_id),
            with_payload=True,
        )
        return _hits(response)

    async def search_hybrid(
        self,
        vector: list[float],
        text: str,
        *,
        limit: int = 5,
        document_id: str | None = None,
        prefetch_limit: int = HYBRID_PREFETCH,
    ) -> list[Hit]:
        """Hybrid: the top ``prefetch_limit`` of both searches merged by Reciprocal Rank
        Fusion inside Qdrant (score = sum of 1 / (60 + rank) over the lists)."""
        sparse = encode_query(text)
        if not sparse.indices:
            return await self.search(vector, limit=limit, document_id=document_id)
        query_filter = _filter(document_id)
        response = await self._client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(
                    query=vector, using=DENSE, limit=prefetch_limit, filter=query_filter
                ),
                models.Prefetch(
                    query=sparse, using=SPARSE, limit=prefetch_limit, filter=query_filter
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return _hits(response)

    async def delete_document(self, document_id: str) -> None:
        await self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def document_names(self) -> list[str]:
        """Every distinct ``document_name`` in the collection, sorted."""
        names: set[str] = set()
        offset = None
        while True:
            points, offset = await self._client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["document_name"],
                with_vectors=False,
            )
            names.update(str(p.payload["document_name"]) for p in points if p.payload)
            if offset is None:
                break
        return sorted(names)

    async def point_count(self) -> int:
        result = await self._client.count(self.collection, exact=True)
        return int(result.count)

    async def exists(self) -> bool:
        return bool(await self._client.collection_exists(self.collection))

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()


def _filter(document_id: str | None) -> models.Filter | None:
    if not document_id:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )


def _hits(response: Any) -> list[Hit]:
    return [
        Hit(
            score=float(p.score),
            text=str(p.payload["text"]),
            document_id=str(p.payload["document_id"]),
            document_name=str(p.payload["document_name"]),
            chunk_index=int(p.payload["chunk_index"]),
        )
        for p in response.points
        if p.payload
    ]
