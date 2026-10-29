"""The text chat (``chat_agent``): the same four steps a call runs, with fakes.

Sarvam and the knowledge store are never contacted; the scripted replies decide
whether the chat searches, so these tests check the wiring of the steps rather
than the model's judgement.
"""

from __future__ import annotations

import json

import httpx
import pytest

from chat_agent.session import ChatSession, ChatSessions
from voice_agent.knowledge.service import SearchResult
from voice_agent.knowledge.store import Hit
from voice_agent.tracking.config_store import AgentConfig

CONFIG = AgentConfig(
    id=7,
    system_prompt="PERSONA.",
    router_prompt="ROUTER RULES.",
    answer_prompt="ANSWER RULES.",
    topic_prompt="CLASSIFIER RULES.",
    language="ta-IN",
    temperature=0.3,
)
QUESTION = "புது ஸ்மார்ட் கார்டு அப்ளை பண்றது எப்படி?"


class FakeKnowledge:
    """Stand-in for ``KnowledgeService``: fixed hits, no embedding, no Qdrant."""

    query_rewrite = False
    qdrant_url = "http://fake-qdrant"

    def __init__(self, *scores: float) -> None:
        self.hits = [
            Hit(score, f"sheet {i} text", f"d{i}", f"{i:03d} - Sheet.pdf", 0)
            for i, score in enumerate(scores)
        ]
        self.asked: list[str] = []

    async def search(self, question, *, limit=5, document_id=None):
        self.asked.append(question)
        return SearchResult(self.hits[:limit], 1, 2)


class FakeSarvam:
    """Scripted chat completions. ``replies`` feed the agent's LLM calls in order."""

    def __init__(self, *replies: dict, classifications: tuple[str, ...] = ()) -> None:
        self.replies = list(replies)
        self.classifications = list(classifications)
        self.bodies: list[dict] = []

    @property
    def llm_bodies(self) -> list[dict]:
        """The agent's own requests, without the topic classifier's."""
        return [b for b in self.bodies if b.get("max_tokens") != 40]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        test = self

        class FakeResponse:
            def __init__(self, data: dict) -> None:
                self._data = data

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self._data

        class FakeClient:
            def __init__(self, *a, **k) -> None: ...

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, *, headers=None, json=None):
                test.bodies.append(json or {})
                return FakeResponse(test._answer(json or {}))

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    def _answer(self, body: dict) -> dict:
        if body.get("max_tokens") == 40:  # the NEW / FOLLOW_UP classifier
            return {
                "choices": [{"message": {"content": self.classifications.pop(0)}}],
                "usage": {},
            }
        reply = self.replies.pop(0)
        message: dict = {"content": reply.get("text")}
        if "questions" in reply:
            message["tool_calls"] = [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": json.dumps(
                            {"questions": reply["questions"]}, ensure_ascii=False
                        ),
                    },
                }
            ]
        return {
            "choices": [
                {
                    "message": message,
                    "finish_reason": "tool_calls" if "questions" in reply else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def chat(settings, knowledge=None, config=CONFIG) -> ChatSession:
    return ChatSession(settings=settings, config=config, knowledge=knowledge)


def systems(sarvam: FakeSarvam) -> list[str]:
    return [body["messages"][0]["content"] for body in sarvam.llm_bodies]


# ---------- the four steps ----------


async def test_each_step_gets_its_own_prompt_and_llm2_writes_the_reply(settings, monkeypatch):
    sarvam = FakeSarvam({"questions": [QUESTION]}, {"text": "ஆன்லைன்ல அப்ளை பண்ணலாம்."})
    sarvam.install(monkeypatch)
    knowledge = FakeKnowledge(0.71, 0.55)
    session = chat(settings, knowledge)

    turn = await session.say(QUESTION)

    assert turn.reply == "ஆன்லைன்ல அப்ளை பண்ணலாம்."  # LLM 2's text, not LLM 1's
    assert [s["step"] for s in turn.steps] == ["llm decide", "search", "llm answer"]
    decide, answer = sarvam.llm_bodies
    # step 2 gets the router rules and the search tool; step 4 gets the answer rules
    assert "ROUTER RULES." in decide["messages"][0]["content"]
    assert "ANSWER RULES." not in decide["messages"][0]["content"]
    assert decide["tools"][0]["function"]["name"] == "search_knowledge_base"
    assert "ANSWER RULES." in answer["messages"][0]["content"]
    assert "PERSONA." in answer["messages"][0]["content"]  # the persona is shared
    # LLM 2 answers a context that ends with the search results
    assert answer["messages"][-1]["role"] == "tool"
    result = json.loads(answer["messages"][-1]["content"])
    assert result["found"] is True
    assert [p["source"] for p in result["passages"]] == ["000 - Sheet.pdf", "001 - Sheet.pdf"]
    assert knowledge.asked == [QUESTION]  # searched in the caller's own words


async def test_a_turn_that_needs_no_facts_never_searches(settings, monkeypatch):
    sarvam = FakeSarvam({"text": "கண்டிப்பா, சொல்லுங்க."})
    sarvam.install(monkeypatch)
    session = chat(settings, FakeKnowledge(0.9))

    turn = await session.say("நன்றி")

    assert turn.reply == "கண்டிப்பா, சொல்லுங்க."
    assert turn.searched is False and turn.searches == []
    assert [s["step"] for s in turn.steps] == ["llm decide"]
    assert len(sarvam.llm_bodies) == 1  # no second call: LLM 1's reply is the answer


async def test_text_written_alongside_a_tool_call_is_dropped(settings, monkeypatch):
    sarvam = FakeSarvam(
        {"text": "ஒரு நிமிஷம் இருங்க…", "questions": [QUESTION]},
        {"text": "ஆன்லைன்ல அப்ளை பண்ணலாம்."},
    )
    sarvam.install(monkeypatch)
    session = chat(settings, FakeKnowledge(0.8))

    turn = await session.say(QUESTION)

    # exactly as on a call: only the reply written after the search reaches the caller
    assert turn.dropped_text == "ஒரு நிமிஷம் இருங்க…"
    assert turn.reply == "ஆன்லைன்ல அப்ளை பண்ணலாம்."
    assert "ஒரு நிமிஷம்" not in str(session.messages)


async def test_a_search_is_announced_with_a_filler_line(settings, monkeypatch):
    sarvam = FakeSarvam({"questions": [QUESTION]}, {"text": "ஆன்லைன்ல."})
    sarvam.install(monkeypatch)
    session = chat(settings, FakeKnowledge(0.8))

    turn = await session.say(QUESTION)

    assert turn.filler  # the line a caller would hear while the search runs
    assert turn.searched is True


# ---------- the score threshold ----------


async def test_nothing_above_the_score_cut_is_not_sent(make_settings, monkeypatch):
    settings = make_settings(knowledge_min_score=0.40)
    sarvam = FakeSarvam({"questions": [QUESTION]}, {"text": "அந்த தகவல் என்கிட்ட இல்ல."})
    sarvam.install(monkeypatch)
    session = chat(settings, FakeKnowledge(0.39, 0.30))

    turn = await session.say(QUESTION)

    result = json.loads(sarvam.llm_bodies[-1]["messages"][-1]["content"])
    assert result["found"] is False and "passages" not in result
    search = next(s for s in turn.steps if s["step"] == "search")
    assert search["found"] is False and search["sheets_sent"] == 0
    assert search["sheets_retrieved"] == 2  # both were retrieved, neither was sent
    assert [h["used"] for h in turn.searches[0]["hits"]] == [False, False]


async def test_the_cut_follows_the_setting(make_settings, monkeypatch):
    settings = make_settings(knowledge_min_score=0.40)
    sarvam = FakeSarvam({"questions": [QUESTION]}, {"text": "சரி."})
    sarvam.install(monkeypatch)
    session = chat(settings, FakeKnowledge(0.45, 0.39))

    await session.say(QUESTION)

    result = json.loads(sarvam.llm_bodies[-1]["messages"][-1]["content"])
    assert [p["score"] for p in result["passages"]] == [0.45]  # 0.39 stays behind


# ---------- topic context ----------


async def test_the_topic_check_runs_first_and_a_new_topic_restarts_the_context(
    make_settings, monkeypatch
):
    settings = make_settings(llm_topic_context=True)
    sarvam = FakeSarvam(
        {"text": "ஆன்லைன்ல அப்ளை பண்ணலாம்."},
        {"text": "பேரை நீக்க படிவம் கொடுக்கணும்."},
        classifications=("NEW: a different request",),
    )
    sarvam.install(monkeypatch)
    session = chat(settings, config=AgentConfig(**{**CONFIG.__dict__, "greeting": "வணக்கம்."}))

    first = await session.say(QUESTION)
    second = await session.say("என்னோட பேரை நீக்கணும்")

    assert [s["step"] for s in first.steps][0] == "topic check"
    assert first.topic == {
        "turn": 1,
        "decision": "NEW",
        "reason": "first question of the call",
        "text": QUESTION,
    }
    assert second.topic and second.topic["decision"] == "NEW"
    assert next(s for s in second.steps if s["step"] == "topic check")["restarted"] is True
    # the second turn's context keeps the greeting but drops the first topic
    sent = [m.get("content") for m in sarvam.llm_bodies[-1]["messages"][1:]]
    assert sent == ["வணக்கம்.", "என்னோட பேரை நீக்கணும்"]
    assert QUESTION in str(session.messages)  # the chat itself keeps everything


async def test_a_follow_up_keeps_the_context(make_settings, monkeypatch):
    settings = make_settings(llm_topic_context=True)
    sarvam = FakeSarvam(
        {"text": "ஆன்லைன்ல அப்ளை பண்ணலாம்."},
        {"text": "ஃபீஸ் ஐம்பது ரூபாய்."},
        classifications=("FOLLOW_UP: asks the fee",),
    )
    sarvam.install(monkeypatch)
    session = chat(settings)

    await session.say(QUESTION)
    await session.say("அதுக்கு எவ்வளவு காசு?")

    sent = [m.get("content") for m in sarvam.llm_bodies[-1]["messages"][1:]]
    assert sent == [QUESTION, "ஆன்லைன்ல அப்ளை பண்ணலாம்.", "அதுக்கு எவ்வளவு காசு?"]


async def test_a_failed_topic_check_keeps_the_context(make_settings, monkeypatch):
    settings = make_settings(llm_topic_context=True)
    sarvam = FakeSarvam({"text": "ஒன்னு."}, {"text": "ரெண்டு."}, classifications=())
    sarvam.install(monkeypatch)
    session = chat(settings)

    await session.say(QUESTION)
    turn = await session.say("இன்னொரு கேள்வி")  # no classification scripted: the call fails

    check = next(s for s in turn.steps if s["step"] == "topic check")
    assert check["decision"] == "FOLLOW_UP" and "classifier failed" in (check["reason"] or "")
    assert turn.reply == "ரெண்டு."  # the turn still answers


# ---------- the chat itself ----------


async def test_the_chat_opens_with_the_configured_greeting(settings):
    session = chat(settings, config=AgentConfig(**{**CONFIG.__dict__, "greeting": "வணக்கம்."}))
    assert session.greeting == "வணக்கம்."
    assert session.messages == [{"role": "assistant", "content": "வணக்கம்."}]
    assert session.as_dict()["language"] == "ta-IN"
    assert session.as_dict()["config_id"] == 7


async def test_without_knowledge_there_is_no_search_tool(settings, monkeypatch):
    sarvam = FakeSarvam({"text": "சரி."})
    sarvam.install(monkeypatch)
    session = chat(settings, knowledge=None)

    turn = await session.say("வணக்கம்")

    assert session.tools is None and session.tool is None
    assert "tools" not in sarvam.llm_bodies[0]
    assert "ROUTER RULES." not in sarvam.llm_bodies[0]["messages"][0]["content"]
    assert turn.reply == "சரி."


async def test_an_empty_reply_is_reported_not_spoken(settings, monkeypatch):
    sarvam = FakeSarvam({"text": ""})
    sarvam.install(monkeypatch)
    session = chat(settings, FakeKnowledge(0.8))

    turn = await session.say("வணக்கம்")

    assert turn.reply == "" and turn.error == "the model returned no text"
    assert session.messages[-1]["role"] == "user"  # nothing empty is added to the chat


async def test_sessions_are_kept_by_id_and_dropped_on_end(settings):
    sessions = ChatSessions()
    session = sessions.start(settings=settings, config=CONFIG, knowledge=None)
    assert sessions.get(session.id) is session
    assert sessions.end(session.id) is True
    assert sessions.get(session.id) is None and sessions.end(session.id) is False


async def test_sessions_drop_the_least_recently_used_over_the_limit(settings):
    sessions = ChatSessions(limit=2)
    first = sessions.start(settings=settings, config=CONFIG, knowledge=None)
    second = sessions.start(settings=settings, config=CONFIG, knowledge=None)
    third = sessions.start(settings=settings, config=CONFIG, knowledge=None)
    assert sessions.get(first.id) is None  # the oldest made room
    assert sessions.get(second.id) is second and sessions.get(third.id) is third
