from voice_agent.languages import get_language
from voice_agent.prompts import build_system_prompt


def test_default_prompt_mentions_name_and_language():
    prompt = build_system_prompt("Asha", get_language("ta-IN"))
    assert "Asha" in prompt
    assert "Tamil" in prompt
    assert "தமிழ்" in prompt


def test_override_prompt_supports_placeholders():
    prompt = build_system_prompt(
        "Meera", get_language("hi-IN"), override="Hi {agent_name}, speak {language_english_name}."
    )
    assert prompt == "Hi Meera, speak Hindi."
