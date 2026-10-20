"""Call tracking and agent configuration stored in Postgres.

Design rule: nothing in this package may block the audio pipeline. Writes are
queued and flushed by a background task (:mod:`recorder`), and the single read
at session start (:mod:`config_store`) is bounded by a short timeout with an
environment fallback. If the database is unreachable the agent keeps running
without tracking.
"""

from __future__ import annotations

import asyncio

import asyncpg
from loguru import logger

from voice_agent.tracking.db import create_pool, ensure_schema

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()
_disabled = False


async def get_tracking_pool(database_url: str | None) -> asyncpg.Pool | None:
    """Return the shared connection pool, creating it on first use.

    Returns ``None`` when tracking is not configured or the database could not
    be reached; a failed attempt is not retried for the life of the process so
    sessions never pay the connection timeout twice.
    """
    global _pool, _disabled
    if database_url is None or _disabled:
        return None
    async with _pool_lock:
        if _pool is None:
            try:
                _pool = await create_pool(database_url)
                await ensure_schema(_pool)
                logger.info("Call tracking enabled (Postgres)")
            except Exception as exc:
                _disabled = True
                logger.warning(f"Call tracking disabled: cannot reach Postgres ({exc})")
                return None
    return _pool
