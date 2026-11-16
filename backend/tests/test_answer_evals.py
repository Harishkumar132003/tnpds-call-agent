"""Answer-prompt evaluation: a frozen conversation + sheets answered with one prompt."""

from __future__ import annotations

import json

from portal.answer_evals import (
    TOOL_CALL_ID,
    answer_item,
    answer_messages,
    fetch_sheets,
    prompt_source_of,
    tool_result,
)
from tests.test_chat import FakeKnowledge
from voice_agent.knowledge.tool import ANSWER_PROMPT, TOOL_NAME
from voice_agent.tracking.config_store import AgentConfig

ITEM = {
    "id": 7,
    "kind": "follow_up",
    "messages": [
        {"role": "user", "text": "என் கார்டு தொலைஞ்சு போச்சு."},
        {"role": "assistant", "text": "புரியுது, ரீபிரின்ட் அப்ளை பண்ணலாம்."},
        {"role": "user", "text": "அதுக்கு எவ்வளவு ஆகும்?"},
    ],
    "rewritten_query": "My card is lost. How much does it cost to get it again?",
    "sheets": [
        {"source": "pb132-1 Reprint smart card", "score": 0.61, "used": True, "text": "Fee: 50."},
        {"source": "pb10 Complaint", "score": 0.30, "used": False, "text": "Complaints."},
    ],
}


def test_only_used_sheets_reach_the_model_as_passages():
    result = tool_result(ITEM)
    assert result["found"] is True
    assert [p["source"] for p in result["passages"]] == ["pb132-1 Reprint smart card"]
    assert result["questions"][0]["question"] == "அதுக்கு எவ்வளவு ஆகும்?"
    assert result["questions"][0]["query_in_english"] == ITEM["rewritten_query"]
    assert tool_result({**ITEM, "sheets": []})["found"] is False


def test_request_is_persona_conversation_search_call_and_result():
    messages = answer_messages(ITEM, "SYSTEM")
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    call = messages[-2]["tool_calls"][0]
    assert call["id"] == TOOL_CALL_ID and call["function"]["name"] == TOOL_NAME
    assert json.loads(call["function"]["arguments"]) == {"questions": ["அதுக்கு எவ்வளவு ஆகும்?"]}
    assert messages[-1]["tool_call_id"] == TOOL_CALL_ID
    assert json.loads(messages[-1]["content"])["passages"][0]["text"] == "Fee: 50."


def test_prompt_source_names_default_config_and_draft():
    config = AgentConfig(answer_prompt="Answer briefly.")
    assert prompt_source_of(ANSWER_PROMPT, config) == "default"
    assert prompt_source_of("Answer briefly.", config) == "config"
    assert prompt_source_of("Something else", config) == "draft"


class _Reply:
    def __init__(self, text: str) -> None:
        self.text, self.prompt_tokens, self.completion_tokens = text, 10, 5


class _Session:
    def __init__(self, text: str | None) -> None:
        self.text, self.requests = text, []

    async def _complete(self, messages, *, tools):
        self.requests.append((messages, tools))
        if self.text is None:
            raise RuntimeError("boom")
        return _Reply(self.text)


async def test_answer_item_records_the_reply_or_the_error():
    session = _Session("ஐம்பது ரூபாய்.")
    out = await answer_item(ITEM, session=session, system="S")
    assert out.answer == "ஐம்பது ரூபாய்." and out.error is None and out.prompt_tokens == 10
    assert session.requests[0][1] is None  # no tools: the answer step cannot search again
    failed = await answer_item(ITEM, session=_Session(None), system="S")
    assert failed.answer is None and failed.error == "boom"
    empty = await answer_item(ITEM, session=_Session("  "), system="S")
    assert empty.error == "empty reply"


async def test_fetch_sheets_marks_what_the_tool_would_send():
    knowledge = FakeKnowledge(0.71, 0.55, 0.30)
    sheets, query = await fetch_sheets(
        ITEM["messages"], service=knowledge, rewrite_prompt=None, limit=5, min_score=0.4
    )
    assert knowledge.asked == ["அதுக்கு எவ்வளவு ஆகும்?"]  # no rewriter: the question itself
    assert [s["used"] for s in sheets] == [True, True, False]
    assert all("text" in s and "source" in s for s in sheets)
    assert query is None


def test_a_score_floor_keeps_the_best_sheet_even_when_none_reach_it():
    item = {
        **ITEM,
        "sheets": [
            {"source": "pb1 A", "score": 0.9, "used": True, "text": "a"},
            {"source": "pb2 B", "score": 0.45, "used": True, "text": "b"},
            {"source": "pb3 C", "score": 0.3, "used": True, "text": "c"},
        ],
    }
    assert [p["source"] for p in tool_result(item, min_score=0.4)["passages"]] == ["pb1 A", "pb2 B"]
    assert [p["source"] for p in tool_result(item, min_score=0.95)["passages"]] == ["pb1 A"]
    assert [p["source"] for p in tool_result(item, max_sheets=1, min_score=0.4)["passages"]] == [
        "pb1 A"
    ]
