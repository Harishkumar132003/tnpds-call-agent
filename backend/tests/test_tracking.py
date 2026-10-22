"""Unit tests for call tracking; the database is faked so nothing blocks."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    LatencyContribution,
    LatencyOwnerKind,
    TTFBBreakdownMetrics,
    UserBotLatencyObserver,
)
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    UserTurnMessageAddedMessage,
)

from voice_agent.tracking.config_store import AgentConfig, load_active_config
from voice_agent.tracking.conversation import ConversationTracker
from voice_agent.tracking.recorder import CallInfo, CallRecorder, TurnRecord


class FakePool:
    """Records every SQL statement; optionally blocks to simulate a stalled database."""

    def __init__(self, *, stall: asyncio.Event | None = None):
        self.statements: list[tuple[str, tuple]] = []
        self._stall = stall

    @asynccontextmanager
    async def acquire(self):
        if self._stall is not None:
            await self._stall.wait()
        conn = AsyncMock()
        conn.execute.side_effect = lambda sql, *args: self.statements.append((sql, args))
        yield conn


CALL = CallInfo(transport="webrtc", language="hi-IN", direction="web")


async def test_recorder_writes_call_turn_and_end_in_order():
    pool = FakePool()
    recorder = CallRecorder(pool, CALL)
    recorder.start()
    recorder.turn_completed(TurnRecord(turn_index=0, assistant_text="hi", response_latency_ms=900))
    await recorder.close("completed")

    kinds = [sql.split()[0:3] for sql, _ in pool.statements]
    assert kinds == [
        ["insert", "into", "calls"],
        ["insert", "into", "turns"],
        ["update", "calls", "set"],
    ]
    assert recorder.stats.written == 3 and recorder.stats.dropped == 0
    _, turn_args = pool.statements[1]
    assert turn_args[0] == recorder.id and turn_args[3] == "hi" and turn_args[7] == 900


async def test_recorder_never_blocks_when_database_stalls():
    stall = asyncio.Event()  # never set: every write hangs
    recorder = CallRecorder(FakePool(stall=stall), CALL, max_queue=2, flush_timeout=0.1)
    recorder.start()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    for i in range(10):
        recorder.turn_completed(TurnRecord(turn_index=i))
    assert loop.time() - t0 < 0.05, "turn_completed must return immediately"
    assert recorder.stats.dropped > 0
    await recorder.close("completed")  # bounded by flush_timeout


class Events:
    """Minimal stand-in for pipecat objects exposing ``event_handler``."""

    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def decorator(fn):
            self.handlers[name] = fn
            return fn

        return decorator

    async def fire(self, name, *args):
        await self.handlers[name](self, *args)


async def test_conversation_tracker_assembles_turns():
    pool = FakePool()
    recorder = CallRecorder(pool, CALL)
    recorder.start()
    observer, user, assistant = Events(), Events(), Events()
    ConversationTracker(recorder, observer).attach(user, assistant)  # type: ignore[arg-type]

    # greeting (turn 0): connect -> first speech
    await observer.fire("on_first_bot_speech_latency", 2.5)
    await assistant.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="Vanakkam", interrupted=False, timestamp="t"),
    )
    # turn 1: user speaks, latency measured with breakdown, assistant replies
    await user.fire(
        "on_user_turn_message_added", UserTurnMessageAddedMessage(content="hello", timestamp="t")
    )
    await observer.fire("on_latency_measured", 1.2)
    await observer.fire(
        "on_latency_breakdown",
        LatencyBreakdown(
            total_secs=1.2,
            user_turn_secs=0.3,
            contributions=[
                LatencyContribution(
                    key="llm_inference",
                    label="LLM inference",
                    owner="SarvamLLMService#0",
                    owner_kind=LatencyOwnerKind.SERVICE,
                    start_time=0,
                    duration_secs=0.6,
                ),
            ],
            ttfb=[
                TTFBBreakdownMetrics(
                    processor="SarvamSTTService#0", model=None, start_time=0, duration_secs=0.15
                ),
                TTFBBreakdownMetrics(
                    processor="SarvamLLMService#0", model="m", start_time=0, duration_secs=0.6
                ),
                TTFBBreakdownMetrics(
                    processor="SarvamTTSService#0", model=None, start_time=0, duration_secs=0.25
                ),
            ],
        ),
    )
    await assistant.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="How can I help?", interrupted=True, timestamp="t"),
    )
    await recorder.close("completed")

    turns = [args for sql, args in pool.statements if sql.startswith("insert into turns")]
    assert len(turns) == 2
    greeting, reply = turns
    assert (
        greeting[1] == 0
        and greeting[2] is None
        and greeting[3] == "Vanakkam"
        and greeting[7] == 2500
    )
    assert reply[1] == 1 and reply[2] == "hello" and reply[3] == "How can I help?"
    assert reply[7] == 1200 and reply[8] == 300  # latency, release
    assert (reply[9], reply[10], reply[11]) == (150, 600, 250)  # stt/llm/tts ttfb
    assert reply[12] is True  # interrupted
    assert json.loads(reply[13]) == [
        {"key": "llm_inference", "label": "LLM inference", "owner": "SarvamLLMService#0", "ms": 600}
    ]


def test_tracker_registers_on_real_observer_events():
    """The event names we subscribe to must exist on pipecat's observer."""
    observer = UserBotLatencyObserver()
    ConversationTracker(CallRecorder(FakePool(), CALL), observer)


async def test_load_active_config_without_pool_uses_defaults():
    assert await load_active_config(None) == AgentConfig()


async def test_load_active_config_falls_back_when_database_errors():
    class BrokenPool:
        @asynccontextmanager
        async def acquire(self):
            raise ConnectionError("db down")
            yield

    assert await load_active_config(BrokenPool()) == AgentConfig()  # type: ignore[arg-type]


async def test_load_active_config_times_out_quickly(monkeypatch):
    import voice_agent.tracking.config_store as cs

    monkeypatch.setattr(cs, "CONFIG_READ_TIMEOUT_SECS", 0.05)

    class SlowPool:
        @asynccontextmanager
        async def acquire(self):
            conn = AsyncMock()

            async def slow(*_):
                await asyncio.sleep(5)

            conn.fetchrow.side_effect = slow
            yield conn

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    assert await load_active_config(SlowPool()) == AgentConfig()  # type: ignore[arg-type]
    assert loop.time() - t0 < 1


def test_portal_config_parsing():
    from portal.app import parse_config

    assert parse_config("", None) == AgentConfig()
    assert parse_config("Hi {agent_name}", 0.4) == AgentConfig(
        system_prompt="Hi {agent_name}", temperature=0.4
    )
    with pytest.raises(ValueError, match="placeholder"):
        parse_config("Hi {who}", None)
    with pytest.raises(ValueError, match="between"):
        parse_config("", 3)


def test_portal_config_language_validation():
    from portal.app import parse_config

    assert parse_config("", None, "ta-IN").language == "ta-IN"
    assert parse_config("", None, "").language is None
    with pytest.raises(ValueError, match="Supported languages"):
        parse_config("", None, "fr-FR")


def test_apply_language_override(make_settings):
    from voice_agent.session import apply_language_override

    base = make_settings(agent_language="hi-IN")
    assert apply_language_override(base, None) is base
    assert apply_language_override(base, "hi-IN") is base
    assert apply_language_override(base, "ta-IN").agent_language == "ta-IN"
    assert apply_language_override(base, "fr-FR").agent_language == "hi-IN"  # ignored, logged


async def test_tool_call_round_does_not_produce_an_empty_turn():
    pool = FakePool()
    recorder = CallRecorder(pool, CALL)
    recorder.start()
    observer, user, assistant = Events(), Events(), Events()
    ConversationTracker(recorder, observer).attach(user, assistant)  # type: ignore[arg-type]
    await user.fire(
        "on_user_turn_message_added", UserTurnMessageAddedMessage(content="q", timestamp="t")
    )
    await assistant.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="", interrupted=False, timestamp="t"),
    )
    await observer.fire("on_latency_measured", 2.0)
    await assistant.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="answer", interrupted=False, timestamp="t"),
    )
    await recorder.close("completed")
    turns = [a for sql, a in pool.statements if sql.startswith("insert into turns")]
    assert (
        len(turns) == 1 and turns[0][2] == "q" and turns[0][3] == "answer" and turns[0][7] == 2000
    )


async def test_tracker_keeps_searches_with_the_turn():
    from voice_agent.knowledge.tool import KnowledgeTool, SearchRecord

    pool = FakePool()
    recorder = CallRecorder(pool, CALL)
    recorder.start()
    observer, user, assistant = Events(), Events(), Events()
    tracker = ConversationTracker(recorder, observer)
    tracker.attach(user, assistant)  # type: ignore[arg-type]
    tool = KnowledgeTool(service=None)  # type: ignore[arg-type]
    tracker.attach_knowledge(tool)

    await user.fire(
        "on_user_turn_message_added", UserTurnMessageAddedMessage(content="q", timestamp="t")
    )
    await tool._notify(
        SearchRecord(
            "q", "q in english", 500, 300, 9, [{"source": "a.pdf", "score": 0.6, "used": True}]
        )
    )
    await assistant.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="a", interrupted=False, timestamp="t"),
    )
    await recorder.close("completed")
    [turn] = [a for sql, a in pool.statements if sql.startswith("insert into turns")]
    searches = json.loads(turn[14])
    assert (
        searches[0]["translated_query"] == "q in english"
        and searches[0]["hits"][0]["source"] == "a.pdf"
    )


async def test_usage_observer_attributes_tokens_to_the_turn():
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTSUsageMetricsData
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    pool = FakePool()
    recorder = CallRecorder(pool, CALL)
    recorder.start()
    observer, user, assistant = Events(), Events(), Events()
    tracker = ConversationTracker(recorder, observer)
    tracker.attach(user, assistant)  # type: ignore[arg-type]
    usage = tracker.usage_observer()
    frame = MetricsFrame(
        data=[
            LLMUsageMetricsData(
                processor="llm",
                model="m",
                value=LLMTokenUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150),
            ),
            TTSUsageMetricsData(processor="tts", model="t", value=48),
        ]
    )
    push = FramePushed(
        source=None, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0
    )  # type: ignore[arg-type]
    for _ in range(4):  # the same frame is seen at every hop downstream of its source
        await usage.on_push_frame(push)
    await assistant.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="hi", interrupted=False, timestamp="t"),
    )
    await recorder.close("completed")
    [turn] = [a for sql, a in pool.statements if sql.startswith("insert into turns")]
    assert (turn[15], turn[16], turn[17]) == (120, 30, 48)


class FakeLangfuse:
    """Records the v2 SDK calls the exporter makes."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.flushed = False

    def trace(self, **kw):
        self.calls.append(("trace", kw))
        return self

    def span(self, **kw):
        self.calls.append(("span", kw))
        return self

    def generation(self, **kw):
        self.calls.append(("generation", kw))
        return self

    def update(self, **kw):
        self.calls.append(("update", kw))

    def flush(self):
        self.flushed = True


async def test_langfuse_exporter_builds_trace_with_usage(make_settings):
    import uuid

    from voice_agent.tracking.langfuse_export import LangfuseExporter

    settings = make_settings(
        langfuse_host="http://x", langfuse_public_key="pk", langfuse_secret_key="sk"
    )
    fake = FakeLangfuse()
    call_id = uuid.uuid4()
    exporter = LangfuseExporter(settings, CALL, call_id, client=fake)
    exporter.start()
    exporter.turn_completed(
        TurnRecord(
            turn_index=1,
            user_text="q",
            assistant_text="a",
            response_latency_ms=900,
            prompt_tokens=100,
            completion_tokens=20,
            tts_characters=40,
            stt_audio_seconds=2.5,
            searches=[{"question": "q", "hits": [{"source": "d", "used": True}]}],
        )
    )
    await exporter.close("completed")
    kinds = [k for k, _ in fake.calls]
    assert kinds == ["trace", "span", "generation", "span", "generation", "generation", "update"]
    assert fake.calls[0][1]["id"] == str(call_id)
    llm = next(kw for k, kw in fake.calls if k == "generation" and kw["name"].endswith("llm"))
    assert (
        llm["usage"] == {"input": 100, "output": 20, "total": 120, "unit": "TOKENS"}
        and llm["model"] == "sarvam-105b"
    )
    assert fake.flushed


async def test_multiple_user_messages_before_a_reply_are_kept_together():
    pool = FakePool()
    recorder = CallRecorder(pool, CALL)
    recorder.start()
    observer, user, assistant = Events(), Events(), Events()
    ConversationTracker(recorder, observer).attach(user, assistant)  # type: ignore[arg-type]
    for part in ("I got married ten days ago.", "I need a new card.", "How do I remove my name?"):
        await user.fire(
            "on_user_turn_message_added", UserTurnMessageAddedMessage(content=part, timestamp="t")
        )
    await assistant.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="Sure.", interrupted=False, timestamp="t"),
    )
    await recorder.close("completed")
    [turn] = [a for sql, a in pool.statements if sql.startswith("insert into turns")]
    assert turn[2] == "I got married ten days ago. I need a new card. How do I remove my name?"


async def test_langfuse_exporter_skips_llm_generation_for_fixed_lines(make_settings):
    import uuid

    from voice_agent.tracking.langfuse_export import LangfuseExporter

    settings = make_settings(
        langfuse_host="http://x", langfuse_public_key="pk", langfuse_secret_key="sk"
    )
    fake = FakeLangfuse()
    exporter = LangfuseExporter(settings, CALL, uuid.uuid4(), client=fake)
    exporter.start()
    exporter.turn_completed(
        TurnRecord(turn_index=1, assistant_text="still there?", tts_characters=12)
    )
    await exporter.close("idle_timeout")
    names = [kw.get("name") for k, kw in fake.calls if k == "generation"]
    assert names == ["turn 1 tts"]


async def test_langfuse_exporter_costs_translation_per_character(make_settings):
    import uuid

    from voice_agent.tracking.langfuse_export import LangfuseExporter

    settings = make_settings(
        langfuse_host="http://x", langfuse_public_key="pk", langfuse_secret_key="sk"
    )
    fake = FakeLangfuse()
    exporter = LangfuseExporter(settings, CALL, uuid.uuid4(), client=fake)
    exporter.start()
    question = "ரேஷன் கார்டு புதுப்பிக்க என்ன வேண்டும்?"
    exporter.turn_completed(
        TurnRecord(
            turn_index=2,
            user_text=question,
            assistant_text="a",
            searches=[
                {
                    "question": question,
                    "translated_query": "What is needed?",
                    "hits": [],
                },
                {"question": "already English", "translated_query": None, "hits": []},
            ],
        )
    )
    await exporter.close("completed")
    [gen] = [kw for k, kw in fake.calls if k == "generation" and kw["name"].endswith("translate")]
    assert gen["model"] == "sarvam-translate:v1"
    assert gen["usage"] == {"input": len(question), "unit": "CHARACTERS"}
    assert gen["output"] == "What is needed?"
    assert not [kw for k, kw in fake.calls if k == "generation" and "embedding" in kw["name"]]
