"""Text embeddings via OpenAI's multilingual embedding models."""

from __future__ import annotations

from typing import Protocol

from openai import AsyncOpenAI

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# Vector sizes of the models we know; others must set EMBEDDING_DIMENSIONS.
KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
BATCH_SIZE = 64


class Embedder(Protocol):
    model: str
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Batches texts through the OpenAI embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        resolved = dimensions or KNOWN_DIMENSIONS.get(model)
        if resolved is None:
            raise ValueError(f"EMBEDDING_DIMENSIONS is required for unknown model {model!r}")
        self.dimensions = resolved
        # Explicit default so a blank OPENAI_BASE_URL in the environment cannot break it.
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url or "https://api.openai.com/v1"
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = [t.replace("\n", " ") for t in texts[start : start + BATCH_SIZE]]
            response = await self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors
