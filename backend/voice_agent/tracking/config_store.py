"""Agent configuration (system prompt, temperature) managed from the portal."""

from __future__ import annotations

import asyncio
import json
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
    idle_prompt_secs: float | None = None
    """Silence before the "still there?" check (and again before hang-up); None = env default."""
    idle_check_line: str | None = None
    idle_goodbye_line: str | None = None
    knowledge_prompt: str | None = None
    """Rules appended to the prompt when the search tool is on; None = built-in text."""
    pronunciations: list[dict[str, str]] | None = None
    """``[{"word", "spoken"}]`` entries of the Sarvam pronunciation dictionary."""
    tts_dict_id: str | None = None
    """Sarvam dictionary id built from ``pronunciations`` (passed to TTS)."""


async def load_active_config(pool: asyncpg.Pool | None) -> AgentConfig:
    """Fetch the newest config version, falling back to defaults on timeout/error."""
    if pool is None:
        return AgentConfig()
    try:
        async with pool.acquire() as conn:
            row = await asyncio.wait_for(
                conn.fetchrow(
                    "select id, system_prompt, temperature, language, greeting, "
                    "idle_prompt_secs, idle_check_line, idle_goodbye_line, knowledge_prompt, "
                    "pronunciations::text as pronunciations, tts_dict_id "
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
        idle_prompt_secs=row["idle_prompt_secs"],
        idle_check_line=row["idle_check_line"],
        idle_goodbye_line=row["idle_goodbye_line"],
        knowledge_prompt=row["knowledge_prompt"],
        pronunciations=json.loads(row["pronunciations"]) if row["pronunciations"] else None,
        tts_dict_id=row["tts_dict_id"],
    )


async def save_config(
    pool: asyncpg.Pool,
    *,
    system_prompt: str | None,
    temperature: float | None,
    note: str | None,
    language: str | None = None,
    greeting: str | None = None,
    idle_prompt_secs: float | None = None,
    idle_check_line: str | None = None,
    idle_goodbye_line: str | None = None,
    knowledge_prompt: str | None = None,
    pronunciations: list[dict[str, str]] | None = None,
    tts_dict_id: str | None = None,
) -> int:
    """Store a new config version and return its id."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "insert into agent_config (system_prompt, temperature, note, language, greeting, "
            "idle_prompt_secs, idle_check_line, idle_goodbye_line, knowledge_prompt, "
            "pronunciations, tts_dict_id) "
            "values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11) returning id",
            system_prompt,
            temperature,
            note,
            language,
            greeting,
            idle_prompt_secs,
            idle_check_line,
            idle_goodbye_line,
            knowledge_prompt,
            json.dumps(pronunciations) if pronunciations is not None else None,
            tts_dict_id,
        )


async def list_config_history(pool: asyncpg.Pool, limit: int = 20) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "select id, system_prompt, temperature, language, greeting, idle_prompt_secs, "
            "idle_check_line, idle_goodbye_line, knowledge_prompt, "
            "pronunciations::text as pronunciations, tts_dict_id, note, created_at "
            "from agent_config "
            "order by id desc limit $1",
            limit,
        )
    history = []
    for row in rows:
        item = dict(row)
        item["pronunciations"] = (
            json.loads(row["pronunciations"]) if row["pronunciations"] else None
        )
        history.append(item)
    return history
