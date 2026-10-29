"""LLM_TOPIC_CONTEXT: new-question detection and topic-only context."""

from __future__ import annotations

from voice_agent.topics import CLASSIFIER_PROMPT, TopicTracker, caller_turn_starts, segment_messages


def user(t):
    return {"role": "user", "content": t}


def agent(t):
    return {"role": "assistant", "content": t}


def call(cid):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function"}]}


def result(cid):
    return {"role": "tool", "tool_call_id": cid, "content": "sheets"}


SYSTEM = {"role": "system", "content": "prompt"}


def texts(messages):
    return [m["content"] for m in messages if m["role"] in ("user", "assistant") and m["content"]]


def test_caller_turn_starts_merge_split_pieces_and_skip_tool_results():
    convo = [
        SYSTEM,
        agent("hi"),
        user("கார்டு தொலைஞ்சு போச்சு."),  # one caller turn, split by STT
        user("என்ன பண்ணனும்?"),
        call("s1"),
        result("s1"),
        agent("reprint..."),
        user("அதுக்கு காசு?"),
    ]
    assert caller_turn_starts(convo) == [2, 7]


def test_segment_keeps_system_and_starts_at_the_topic_turn():
    convo = [
        SYSTEM,
        agent("hi"),
        user("u1"),
        agent("a1"),
        user("u2"),
        call("s"),
        result("s"),
        agent("a2"),
    ]
    assert segment_messages(convo, 1) == convo
    seg = segment_messages(convo, 2)
    assert seg[0] == SYSTEM and texts(seg) == ["hi", "u2", "a2"]  # the greeting is kept
    assert [m.get("tool_call_id") for m in seg if m["role"] == "tool"] == ["s"]
    assert segment_messages(convo, 9) == convo  # beyond the last turn: unchanged


class FakeTracker(TopicTracker):
    """TopicTracker with the Sarvam call replaced by a scripted answer."""

    def __init__(self, answers):
        super().__init__("key")
        self.answers = list(answers)
        self.seen = []

    async def _classify(self, context_lines, latest):
        self.seen.append((context_lines, latest))
        return self.answers.pop(0)


async def test_new_question_restarts_the_context_follow_up_keeps_it():
    tracker = FakeTracker(
        [
            ("FOLLOW_UP", "asks the fee"),
            ("NEW", "different request"),
            ("FOLLOW_UP", "documents for it"),
        ]
    )
    convo = [SYSTEM, agent("hi"), user("கார்டு தொலைஞ்சு போச்சு")]
    await tracker.update(convo)  # first question: NEW without a classifier call
    assert tracker.segment_start == 1 and tracker.seen == []
    assert texts(tracker.apply(convo)) == ["hi", "கார்டு தொலைஞ்சு போச்சு"]

    convo += [agent("reprint, ₹50"), user("அதுக்கு எவ்வளவு காசு?")]
    await tracker.update(convo)
    assert tracker.segment_start == 1  # follow-up: context kept
    assert tracker.seen[-1][1] == "அதுக்கு எவ்வளவு காசு?"
    assert "Caller: கார்டு தொலைஞ்சு போச்சு" in tracker.seen[-1][0]

    convo += [agent("50 rupees"), user("என் பொண்ணு பேரை கார்டுல சேர்க்கணும்")]
    await tracker.update(convo)
    assert tracker.segment_start == 3  # new question: context restarts here
    assert texts(tracker.apply(convo)) == ["hi", "என் பொண்ணு பேரை கார்டுல சேர்க்கணும்"]
    assert tracker.apply(convo)[0]["content"].startswith("prompt")
    assert "Never greet again" in tracker.apply(convo)[0]["content"]  # earlier topic hidden
    assert convo[0] == SYSTEM  # the context itself is not changed

    convo += [agent("add member..."), user("அதுக்கு என்ன டாக்குமெண்ட்ஸ்?")]
    await tracker.update(convo)
    assert tracker.segment_start == 3
    assert texts(tracker.apply(convo)) == [
        "hi",
        "என் பொண்ணு பேரை கார்டுல சேர்க்கணும்",
        "add member...",
        "அதுக்கு என்ன டாக்குமெண்ட்ஸ்?",
    ]
    # the classifier only ever saw the current topic, not the lost-card turns
    assert all("தொலைஞ்சு" not in line for line in tracker.seen[-1][0])


async def test_each_caller_turn_is_classified_once_and_failures_keep_context():
    tracker = FakeTracker([("FOLLOW_UP", "x")])
    convo = [SYSTEM, user("u1"), agent("a1"), user("u2")]
    await tracker.update(convo)
    await tracker.update(convo)  # the answer call of the same turn: no second classification
    assert len(tracker.seen) == 1
    assert [d["decision"] for d in tracker.decisions] == [
        "FOLLOW_UP"
    ]  # turn 2 only; turn 1 was never seen alone

    broken = TopicTracker("key", timeout=0.01)
    broken._classified = 1
    broken.segment_start = 1
    decision, reason = await broken._classify(["Caller: u1"], "u2")
    assert decision == "FOLLOW_UP" and "classifier failed" in reason


async def test_llm_request_carries_only_the_current_topic(settings):
    from pipecat.processors.aggregators.llm_context import LLMContext

    from voice_agent.services import build_llm

    llm = build_llm(settings.model_copy(update={"llm_topic_context": True}), "sys")
    tracker = FakeTracker([("NEW", "different")])
    llm.topics = tracker
    seen = []
    llm.on_request = seen.append
    messages = [user("u1"), agent("a1"), user("u2")]
    await tracker.update(messages)
    context = LLMContext(messages=messages)
    adapter = llm.get_llm_adapter()
    params = llm.build_chat_completion_params(
        adapter.get_llm_invocation_params(
            context, system_instruction="sys", convert_developer_to_user=True
        )
    )
    assert params["messages"][0]["role"] == "system"
    assert texts(params["messages"]) == ["u2"]
    assert texts(seen[0]["messages"]) == ["u2"]  # Langfuse sees what was sent


def test_classifier_call_is_exported_with_its_label(make_settings):
    import uuid

    from tests.test_tracking import CALL, FakeLangfuse
    from voice_agent.tracking.langfuse_export import LangfuseExporter
    from voice_agent.tracking.recorder import TurnRecord

    s = make_settings(langfuse_host="http://x", langfuse_public_key="pk", langfuse_secret_key="sk")
    fake = FakeLangfuse()
    exporter = LangfuseExporter(s, CALL, uuid.uuid4(), client=fake)
    exporter.start()
    exporter.turn_completed(
        TurnRecord(
            turn_index=2,
            user_text="q",
            assistant_text="a",
            llm_calls=[
                {
                    "label": "topic check",
                    "model": "sarvam-105b",
                    "input": [{"role": "system", "content": "c"}],
                    "output": "NEW: different request",
                    "tool_calls": [],
                    "usage": {"input": 300, "output": 6},
                    "started_at": None,
                    "ended_at": None,
                },
            ],
        )
    )
    names = [kw["name"] for k, kw in fake.calls if k == "generation"]
    assert "turn 2 topic check" in names  # named by step, not by position


async def test_cancelled_classification_is_retried_on_the_rerun():
    """Pipecat cancels the LLM request when the caller keeps talking; the turn must not be
    marked as classified until a decision was actually made."""
    import asyncio
    import contextlib

    class CancelOnce(TopicTracker):
        def __init__(self):
            super().__init__("key")
            self.calls = 0

        async def _classify(self, context_lines, latest):
            self.calls += 1
            if self.calls == 1:
                raise asyncio.CancelledError()
            return "NEW", "different request"

    tracker = CancelOnce()
    convo = [SYSTEM, user("u1"), agent("a1"), user("piece one.")]
    await tracker.update([SYSTEM, user("u1")])
    with contextlib.suppress(asyncio.CancelledError):
        await tracker.update(convo)  # first piece: request cancelled mid-classification
    assert tracker.segment_start == 1
    convo.append(user("piece two, the real question"))
    await tracker.update(convo)  # the rerun with the full utterance classifies it
    assert tracker.calls == 2 and tracker.segment_start == 2
    assert tracker.decisions[-1]["text"] == "piece one. piece two, the real question"


async def test_decisions_reach_the_turn_record_and_postgres():
    import json

    from pipecat.processors.aggregators.llm_response_universal import (
        AssistantTurnStoppedMessage,
        UserTurnMessageAddedMessage,
    )

    from tests.test_tracking import CALL, Events, FakePool
    from voice_agent.tracking.conversation import ConversationTracker
    from voice_agent.tracking.recorder import CallRecorder

    pool = FakePool()
    recorder = CallRecorder(pool, CALL)
    recorder.start()
    observer, u, a = Events(), Events(), Events()
    tracker = ConversationTracker(recorder, observer)
    tracker.attach(u, a)  # type: ignore[arg-type]
    topics = FakeTracker([("NEW", "different request")])
    topics.on_decision = tracker.record_topic

    await u.fire(
        "on_user_turn_message_added", UserTurnMessageAddedMessage(content="u1", timestamp="t")
    )
    await topics.update([SYSTEM, user("u1")])
    await a.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="a1", interrupted=False, timestamp="t"),
    )
    await u.fire(
        "on_user_turn_message_added", UserTurnMessageAddedMessage(content="u2", timestamp="t")
    )
    await topics.update([SYSTEM, user("u1"), agent("a1"), user("u2")])
    await a.fire(
        "on_assistant_turn_stopped",
        AssistantTurnStoppedMessage(content="a2", interrupted=False, timestamp="t"),
    )
    await recorder.close("completed")
    inserts = [args for sql, args in pool.statements if sql.startswith("insert into turns")]
    stored = [json.loads(args[-1]) if args[-1] else None for args in inserts]
    assert [t and t["decision"] for t in stored] == ["NEW", "NEW"]
    assert stored[1]["reason"] == "different request" and stored[1]["text"] == "u2"


async def test_classifier_sees_the_whole_current_topic():
    tracker = TopicTracker("key")
    seen = {}

    async def fake_classify(lines, latest):
        seen["lines"] = lines
        return "FOLLOW_UP", "more on the same card"

    tracker._classify = fake_classify
    long_answer = "விவரம் " * 200  # well over 300 characters
    convo = [SYSTEM, {"role": "assistant", "content": "hi"}]
    for i in range(12):
        convo += [user(f"u{i}"), {"role": "assistant", "content": long_answer}]
    tracker._classified, tracker.segment_start = 12, 1
    await tracker.update(convo + [user("அதுக்கு?")])
    assert len(seen["lines"]) == 1 + 24  # greeting + all 12 caller/agent pairs
    assert seen["lines"][-1] == f"Agent: {long_answer.strip()}"  # not cut


async def test_classifier_prompt_can_be_overridden():
    tracker = TopicTracker("key", prompt="My own classifier rules.")
    assert tracker.prompt == "My own classifier rules."
    assert TopicTracker("key").prompt == CLASSIFIER_PROMPT
    assert TopicTracker("key", prompt="  ").prompt == CLASSIFIER_PROMPT  # blank = built-in

    sent = {}

    async def fake_post(url, headers=None, json=None):
        sent["system"] = json["messages"][0]["content"]
        raise RuntimeError("stop here")  # the request itself is not under test

    import httpx

    class FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        post = staticmethod(fake_post)

    original = httpx.AsyncClient
    httpx.AsyncClient = FakeClient
    try:
        await tracker._classify(["Caller: hi"], "latest")
    finally:
        httpx.AsyncClient = original
    assert sent["system"] == "My own classifier rules."
