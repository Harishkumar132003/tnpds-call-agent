from pipecat.processors.aggregators.llm_context import LLMContext

from voice_agent.session import _count_user_turns


def test_count_user_turns_counts_only_user_messages():
    context = LLMContext(
        messages=[
            {"role": "developer", "content": "intro"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
    )
    assert _count_user_turns(context) == 2
    assert _count_user_turns(LLMContext()) == 0
