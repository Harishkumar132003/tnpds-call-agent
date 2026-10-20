"""Real Qdrant round trip with a fake embedder (no OpenAI key needed)."""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from voice_agent.knowledge.chunk import Chunk
from voice_agent.knowledge.store import KnowledgeStore

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
pytestmark = pytest.mark.integration


async def _qdrant_up() -> bool:
    try:
        await AsyncQdrantClient(url=QDRANT_URL, timeout=3).get_collections()
        return True
    except Exception:
        return False


async def test_store_round_trip():
    if not await _qdrant_up():
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL}")
    collection = f"test_{uuid.uuid4().hex[:8]}"
    store = KnowledgeStore(url=QDRANT_URL, api_key=None, collection=collection, vector_size=4)
    try:
        await store.ensure_collection()
        await store.ensure_collection()  # idempotent
        doc_a, doc_b = str(uuid.uuid4()), str(uuid.uuid4())
        await store.upsert(
            doc_a, "a.txt", [Chunk(0, "alpha"), Chunk(1, "beta")], [[1, 0, 0, 0], [0, 1, 0, 0]]
        )
        await store.upsert(doc_b, "b.txt", [Chunk(0, "gamma")], [[0, 0, 1, 0]])
        assert await store.point_count() == 3

        hits = await store.search([1, 0, 0, 0], limit=2)
        assert hits[0].text == "alpha" and hits[0].document_name == "a.txt"
        only_b = await store.search([1, 0, 0, 0], limit=5, document_id=doc_b)
        assert [h.text for h in only_b] == ["gamma"]

        await store.delete_document(doc_a)
        assert await store.point_count() == 1
    finally:
        await store._client.delete_collection(collection)
        await store.close()
