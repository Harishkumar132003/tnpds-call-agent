"""Connection pool and schema for the tracking database."""

from __future__ import annotations

import asyncpg

SCHEMA = """
create table if not exists agent_config (
    id            serial primary key,
    system_prompt text,
    temperature   double precision
                  check (temperature is null or (temperature >= 0 and temperature <= 2)),
    note          text,
    created_at    timestamptz not null default now()
);
alter table agent_config add column if not exists language text;
alter table agent_config add column if not exists greeting text;
alter table agent_config add column if not exists idle_prompt_secs double precision;
alter table agent_config add column if not exists idle_check_line text;
alter table agent_config add column if not exists idle_goodbye_line text;
alter table agent_config add column if not exists knowledge_prompt text;
alter table agent_config add column if not exists router_prompt text;
alter table agent_config add column if not exists answer_prompt text;
alter table agent_config add column if not exists topic_prompt text;
alter table agent_config add column if not exists rewrite_prompt text;
alter table agent_config add column if not exists pronunciations jsonb;
alter table agent_config add column if not exists tts_dict_id text;
alter table agent_config add column if not exists stt_model text;
alter table agent_config add column if not exists llm_model text;
alter table agent_config add column if not exists tts_model text;
alter table agent_config add column if not exists tts_speaker text;
alter table agent_config add column if not exists translate_model text;
alter table agent_config add column if not exists embedding_model text;
alter table documents add column if not exists content text;

create table if not exists calls (
    id            uuid primary key,
    session_id    text,
    transport     text not null,
    direction     text,
    from_number   text,
    to_number     text,
    call_id       text,
    stream_id     text,
    language      text not null,
    config_id     integer references agent_config(id),
    started_at    timestamptz not null,
    ended_at      timestamptz,
    end_reason    text,
    turn_count    integer not null default 0
);
create index if not exists calls_started_at_idx on calls (started_at desc);

create table if not exists turns (
    id                  bigserial primary key,
    call_id             uuid not null references calls(id) on delete cascade,
    turn_index          integer not null,
    user_text           text,
    assistant_text      text,
    user_spoke_at       timestamptz,
    bot_spoke_at        timestamptz,
    completed_at        timestamptz,
    response_latency_ms integer,
    turn_release_ms     integer,
    stt_ttfb_ms         integer,
    llm_ttfb_ms         integer,
    tts_ttfb_ms         integer,
    interrupted         boolean not null default false,
    unique (call_id, turn_index)
);
alter table turns add column if not exists contributions jsonb;
alter table turns add column if not exists searches jsonb;
alter table turns add column if not exists topic jsonb;
alter table turns add column if not exists prompt_tokens integer;
alter table turns add column if not exists completion_tokens integer;
alter table turns add column if not exists tts_characters integer;
alter table turns add column if not exists stt_audio_seconds real;

create table if not exists documents (
    id              uuid primary key,
    name            text not null,
    media_type      text,
    size_bytes      integer,
    chars           integer,
    chunk_count     integer,
    embedding_model text,
    collection      text,
    status          text not null,
    error           text,
    created_at      timestamptz not null default now()
);

-- Evaluation: test questions, each run (batch) and one result row per question per batch.
create table if not exists eval_questions (
    id              bigserial primary key,
    question_en     text not null,
    question_ta     text,
    expected_source text not null,
    note            text,
    enabled         boolean not null default true,
    origin          text not null default 'manual',
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);
create table if not exists eval_batches (
    id          uuid primary key,
    started_at  timestamptz not null default now(),
    finished_at timestamptz,
    status      text not null,
    total       integer not null default 0,
    done        integer not null default 0,
    config_id   integer,
    collection  text,
    note        text,
    error       text
);
alter table eval_batches add column if not exists concurrency integer not null default 1;
alter table eval_batches add column if not exists force_search boolean not null default false;
alter table eval_batches add column if not exists search_type text not null default 'semantic';
create table if not exists rewrite_eval_questions (
    id                 bigserial primary key,
    question_ta        text not null,
    question_en        text,
    expected_source    text not null,
    source_question_id bigint,
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now()
);
create table if not exists rewrite_eval_runs (
    id            uuid primary key,
    started_at    timestamptz not null default now(),
    finished_at   timestamptz,
    status        text not null,
    prompt        text not null,
    prompt_source text not null default 'config',
    config_id     integer,
    model         text,
    collection    text,
    search_type   text not null default 'semantic',
    total         integer not null default 0,
    error         text
);
create table if not exists rewrite_eval_results (
    id              bigserial primary key,
    run_id          uuid not null references rewrite_eval_runs(id) on delete cascade,
    question_id     bigint references rewrite_eval_questions(id) on delete set null,
    question_ta     text not null,
    expected_source text not null,
    rewritten_query text,
    hits            jsonb not null default '[]',
    expected_rank   integer,
    expected_sent   boolean not null default false,
    rewrite_ms      integer,
    ms              integer,
    error           text,
    at              timestamptz not null default now()
);
alter table rewrite_eval_results add column if not exists judged_ok boolean not null default false;
create table if not exists eval_results (
    id                bigserial primary key,
    batch_id          uuid not null references eval_batches(id) on delete cascade,
    question_id       bigint references eval_questions(id) on delete set null,
    question_ta       text,
    expected_source   text,
    rewritten_query   text,
    hits              jsonb,
    expected_rank     integer,
    expected_sent     boolean not null default false,
    answer            text,
    error             text,
    prompt_tokens     integer,
    completion_tokens integer,
    ms                integer,
    at                timestamptz not null default now(),
    unique (batch_id, question_id)
);
"""


async def create_pool(database_url: str) -> asyncpg.Pool:
    """Small pool: the bot writes from one background task per session."""
    return await asyncpg.create_pool(
        database_url, min_size=1, max_size=4, command_timeout=10, timeout=5
    )


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA)
