"""Qdrant storage for document chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from voice_agent.knowledge.chunk import Chunk


@dataclass(frozen=True)
class Hit:
    score: float
    text: str
    document_id: str
    document_name: str
    chunk_index: int


class KnowledgeStore:
    """One Qdrant collection holding every document's chunks, filterable by document."""

    def __init__(self, *, url: str, api_key: str | None, collection: str, vector_size: int) -> None:
        self.collection = collection
        self.vector_size = vector_size
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=30)

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self.collection):
            await self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
            )
            await self._client.create_payload_index(
                self.collection, "document_id", models.PayloadSchemaType.KEYWORD
            )

    async def upsert(
        self, document_id: str, document_name: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.UUID(document_id), str(chunk.index))),
                vector=vector,
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
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        response = await self._client.query_points(
            self.collection, query=vector, limit=limit, query_filter=query_filter, with_payload=True
        )
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

    async def point_count(self) -> int:
        result = await self._client.count(self.collection, exact=True)
        return int(result.count)

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
