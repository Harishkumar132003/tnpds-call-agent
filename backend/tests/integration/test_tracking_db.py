"""Round trip against the real Postgres named by DATABASE_URL (opt-in)."""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from portal.app import create_app
from voice_agent.tracking.config_store import load_active_config, save_config
from voice_agent.tracking.db import create_pool, ensure_schema
from voice_agent.tracking.recorder import CallInfo, CallRecorder, TurnRecord

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set"),
]


async def test_schema_record_and_read_back():
    pool = await create_pool(os.environ["DATABASE_URL"])
    try:
        await ensure_schema(pool)
        recorder = CallRecorder(pool, CallInfo(transport="test", language="hi-IN"))
        recorder.start()
        recorder.turn_completed(
            TurnRecord(turn_index=0, assistant_text="hi", response_latency_ms=800)
        )
        await recorder.close("completed")
        assert recorder.stats.failed == 0 and recorder.stats.written == 3
        async with pool.acquire() as conn:
            call = await conn.fetchrow("select * from calls where id = $1", recorder.id)
            turns = await conn.fetch("select * from turns where call_id = $1", recorder.id)
            await conn.execute("delete from calls where id = $1", recorder.id)
        assert call["turn_count"] == 1 and call["end_reason"] == "completed"
        assert turns[0]["response_latency_ms"] == 800
    finally:
        await pool.close()


async def test_config_versions():
    pool = await create_pool(os.environ["DATABASE_URL"])
    try:
        await ensure_schema(pool)
        marker = f"test-{uuid.uuid4()}"
        new_id = await save_config(
            pool, system_prompt="Say hi {agent_name}", temperature=0.3, note=marker
        )
        active = await load_active_config(pool)
        assert active.id == new_id and active.temperature == pytest.approx(0.3)
        async with pool.acquire() as conn:
            await conn.execute("delete from agent_config where note = $1", marker)
    finally:
        await pool.close()


def test_portal_api():
    app = create_app(os.environ["DATABASE_URL"], default_prompt_language="hi-IN", agent_name="Asha")
    with TestClient(app) as client:
        stats = client.get("/api/stats").json()
        assert set(stats) >= {"calls_total", "avg_latency_ms", "by_transport", "latency_buckets"}
        assert client.get("/api/calls").status_code == 200
        config = client.get("/api/config").json()
        assert config["default_prompt"] and len(config["languages"]) == 11
        preview = client.get("/api/config/preview", params={"language": "ta-IN"}).json()
        assert "Tamil" in preview["prompt"]
        assert client.get("/api/config/history").status_code == 200
        assert client.get(f"/api/calls/{uuid.uuid4()}").status_code == 404
        bad = client.post("/api/config", json={"system_prompt": "Hi {who}"})
        assert bad.status_code == 422
        # SPA routes are served by the frontend (or a 503 pointer when not built)
        assert client.get("/calls").status_code in (200, 503)
