"""Agent configuration (system prompt, temperature) managed from the portal."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import asyncpg
from loguru import logger

CONFIG_READ_TIMEOUT_SECS = 1.0


@dataclass(frozen=True)
class AgentConfig:
    """Overrides for one session. ``None`` means "use the environment default"."""

    id: int | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    language: str | None = None
    """Sarvam language code (e.g. ``ta-IN``) overriding ``AGENT_LANGUAGE``."""
    greeting: str | None = None
    """Fixed opening line spoken immediately via TTS; ``None`` lets the LLM greet."""


async def load_active_config(pool: asyncpg.Pool | None) -> AgentConfig:
    """Fetch the newest config version, falling back to defaults on timeout/error."""
    if pool is None:
        return AgentConfig()
    try:
        async with pool.acquire() as conn:
            row = await asyncio.wait_for(
                conn.fetchrow(
                    "select id, system_prompt, temperature, language, greeting "
                    "from agent_config order by id desc limit 1"
                ),
                timeout=CONFIG_READ_TIMEOUT_SECS,
            )
    except Exception as exc:
        logger.warning(f"Using environment agent config; database read failed: {exc}")
        return AgentConfig()
    if row is None:
        return AgentConfig()
    return AgentConfig(
        id=row["id"],
        system_prompt=row["system_prompt"],
        temperature=row["temperature"],
        language=row["language"],
        greeting=row["greeting"],
    )


async def save_config(
    pool: asyncpg.Pool,
    *,
    system_prompt: str | None,
    temperature: float | None,
    note: str | None,
    language: str | None = None,
    greeting: str | None = None,
) -> int:
    """Store a new config version and return its id."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "insert into agent_config (system_prompt, temperature, note, language, greeting) "
            "values ($1, $2, $3, $4, $5) returning id",
            system_prompt,
            temperature,
            note,
            language,
            greeting,
        )


async def list_config_history(pool: asyncpg.Pool, limit: int = 20) -> list[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(
            "select id, system_prompt, temperature, language, greeting, note, created_at "
            "from agent_config "
            "order by id desc limit $1",
            limit,
        )
