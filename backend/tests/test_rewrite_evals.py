"""Rewrite-prompt evaluation: one question rewritten with a prompt, then searched."""

from __future__ import annotations

from portal.rewrite_evals import RunIn, evaluate_question
from tests.test_chat import FakeKnowledge
from voice_agent.knowledge.rewrite import Rewrite


class FakeRewriter:
    def __init__(self, query: str | None) -> None:
        self.query = query
        self.calls: list[tuple[list[str], str | None]] = []

    async def rewrite(self, caller_messages, prompt=None):
        self.calls.append((caller_messages, prompt))
        return None if self.query is None else Rewrite(self.query)


async def test_question_is_rewritten_with_the_prompt_then_searched_and_ranked():
    rewriter = FakeRewriter("how much rice for 4 members")
    knowledge = FakeKnowledge(0.71, 0.55, 0.30)
    out = await evaluate_question(
        "நாலு பேருக்கு எவ்வளவு அரிசி?",
        "001",
        rewriter=rewriter,
        prompt="PROMPT",
        service=knowledge,
        limit=5,
        min_score=0.40,
    )
    assert rewriter.calls == [(["நாலு பேருக்கு எவ்வளவு அரிசி?"], "PROMPT")]
    assert knowledge.asked == ["how much rice for 4 members"]
    assert out.rewritten_query == "how much rice for 4 members"
    assert out.expected_rank == 2 and out.expected_sent is True and out.error is None
    assert [h["used"] for h in out.hits] == [True, True, False]


async def test_a_failed_rewrite_is_an_error_not_a_fallback():
    knowledge = FakeKnowledge(0.9)
    out = await evaluate_question(
        "q",
        "000",
        rewriter=FakeRewriter(None),
        prompt="P",
        service=knowledge,
        limit=5,
        min_score=0.4,
    )
    assert out.error and knowledge.asked == [] and out.expected_rank is None and out.hits == []


async def test_keyword_mode_sends_every_hit():
    knowledge = FakeKnowledge(9.0, 0.2)
    knowledge.search_type = "keyword"  # type: ignore[attr-defined]
    out = await evaluate_question(
        "q",
        "001",
        rewriter=FakeRewriter("x"),
        prompt="P",
        service=knowledge,
        limit=5,
        min_score=0.4,
    )
    assert out.expected_rank == 2 and out.expected_sent is True


def test_run_body_defaults():
    body = RunIn()
    assert body.prompt is None and body.search_type == "semantic"
