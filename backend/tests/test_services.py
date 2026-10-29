import pytest
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language

from voice_agent.config import ConfigurationError
from voice_agent.services import (
    OpenAICompatibleLLMService,
    build_llm,
    build_stt,
    build_tts,
    validate_service_configuration,
)


def test_build_stt_pins_language(make_settings):
    stt = build_stt(make_settings(agent_language="ta-IN", sarvam_stt_mode="verbatim"))
    assert isinstance(stt, SarvamSTTService)
    assert stt.settings.model == "saaras:v4"
    assert stt.settings.language == Language.TA_IN
    assert stt._get_language_string() == "ta-IN"
    assert stt._mode == "verbatim"
    assert stt._init_sample_rate is None  # resolved from the pipeline (16 kHz) at start


def test_build_stt_auto_detect(make_settings):
    stt = build_stt(make_settings(stt_auto_detect_language=True))
    assert stt._get_language_string() == "unknown"


def test_build_stt_rejects_unknown_model(make_settings):
    with pytest.raises(ValueError, match="Unsupported model"):
        build_stt(make_settings(sarvam_stt_model="saarika:v9"))


def test_build_tts_defaults(settings):
    tts = build_tts(settings)
    assert isinstance(tts, SarvamTTSService)
    assert tts.settings.model == "bulbul:v3"
    assert tts.settings.language == Language.HI_IN
    assert tts.settings.voice == "shubh"  # model default speaker
    assert tts._init_sample_rate == 24000  # bulbul:v3 default


def test_build_tts_custom_speaker_and_pace(make_settings):
    tts = build_tts(make_settings(sarvam_tts_speaker="priya", sarvam_tts_pace=1.2))
    assert tts.settings.voice == "priya"
    assert tts.settings.pace == 1.2


def test_build_tts_rejects_unknown_speaker(make_settings):
    with pytest.raises(ConfigurationError, match="not available"):
        build_tts(make_settings(sarvam_tts_speaker="nobody"))


def test_build_llm_sarvam(settings):
    llm = build_llm(settings, "You are a test bot.")
    assert isinstance(llm, SarvamLLMService)
    assert llm.settings.model == "sarvam-105b"
    assert llm.settings.system_instruction == "You are a test bot."
    assert llm.settings.reasoning_effort == "none"


def test_build_llm_sarvam_rejects_unknown_model(make_settings):
    with pytest.raises(ValueError, match="Unsupported Sarvam LLM model"):
        build_llm(make_settings(llm_model="gpt-4.1-mini"), "prompt")


def test_build_llm_openai_compatible(make_settings):
    llm = build_llm(
        make_settings(
            llm_provider="openai",
            llm_model="llama-3.3-70b-versatile",
            openai_api_key="k",
            openai_base_url="https://api.groq.com/openai/v1",
        ),
        "prompt",
    )
    assert llm.settings.system_instruction == "prompt"
    assert isinstance(llm, OpenAILLMService)
    assert llm.settings.model == "llama-3.3-70b-versatile"
    assert str(llm._client.base_url).rstrip("/") == "https://api.groq.com/openai/v1"


def test_build_tts_rejects_pace_outside_model_range(make_settings):
    with pytest.raises(ConfigurationError, match="SARVAM_TTS_PACE"):
        build_tts(make_settings(sarvam_tts_pace=2.5))  # valid for v2, not for bulbul:v3


def test_build_tts_rejects_unknown_model(make_settings):
    with pytest.raises(ConfigurationError, match="SARVAM_TTS_MODEL"):
        build_tts(make_settings(sarvam_tts_model="bulbul:v9"))


def test_build_llm_openai_without_base_url_keeps_developer_role(make_settings):
    llm = build_llm(make_settings(llm_provider="openai", openai_api_key="k"), "prompt")
    assert type(llm) is OpenAILLMService
    assert llm.supports_developer_role is True


def test_build_llm_openai_compatible_disables_developer_role(make_settings):
    llm = build_llm(
        make_settings(llm_provider="openai", openai_api_key="k", openai_base_url="https://x/v1"),
        "prompt",
    )
    assert isinstance(llm, OpenAICompatibleLLMService)
    assert llm.supports_developer_role is False


def test_validate_service_configuration_passes_for_defaults(settings):
    validate_service_configuration(settings)


def test_validate_service_configuration_reports_bad_model(make_settings):
    with pytest.raises(ConfigurationError, match="Unsupported model"):
        validate_service_configuration(make_settings(sarvam_stt_model="saarika:v9"))


def test_validate_service_configuration_reports_bad_prompt_placeholder(make_settings):
    with pytest.raises(ConfigurationError, match="AGENT_SYSTEM_PROMPT"):
        validate_service_configuration(make_settings(agent_system_prompt="Hello {who}"))


class _Streamed:
    """Stands in for Pipecat's streaming loop: pushes text, then maybe a tool call."""

    def __init__(self, llm, monkeypatch, *, text, tool_call):
        self.spoken, self.tools = [], []

        async def fake_process(_self, context):
            for piece in text:
                await _self._push_llm_text(piece)
            if tool_call:
                await _self.run_function_calls(["search_knowledge_base"])

        async def fake_push(_self, piece):
            self.spoken.append(piece)

        async def fake_run(_self, calls):
            self.tools.extend(calls)

        monkeypatch.setattr(SarvamLLMService, "_process_context", fake_process)
        monkeypatch.setattr(SarvamLLMService, "_push_llm_text", fake_push)
        monkeypatch.setattr(SarvamLLMService, "run_function_calls", fake_run)


def _context(*messages):
    from pipecat.processors.aggregators.llm_context import LLMContext

    return LLMContext(messages=list(messages))


async def test_llm1_text_is_dropped_when_it_calls_the_tool(settings, monkeypatch):
    llm = build_llm(settings, "prompt")
    llm.hold_text_before_tools = True
    run = _Streamed(llm, monkeypatch, text=["ஓகே, ", "செக் பண்ணி சொல்றேன்."], tool_call=True)
    await llm._process_context(_context({"role": "user", "content": "பையன் பேரை சேர்க்கணும்"}))
    assert run.spoken == [] and run.tools == ["search_knowledge_base"]


async def test_llm1_text_is_spoken_when_no_tool_is_called(settings, monkeypatch):
    llm = build_llm(settings, "prompt")
    llm.hold_text_before_tools = True
    run = _Streamed(llm, monkeypatch, text=["சரி, ", "வேற ஏதாவது?"], tool_call=False)
    await llm._process_context(_context({"role": "user", "content": "நன்றி"}))
    assert run.spoken == ["சரி, வேற ஏதாவது?"]


async def test_llm2_reply_after_the_tool_result_streams(settings, monkeypatch):
    llm = build_llm(settings, "prompt")
    llm.hold_text_before_tools = True
    run = _Streamed(llm, monkeypatch, text=["ஆதார் ", "கார்டு வேணும்."], tool_call=False)
    await llm._process_context(
        _context(
            {"role": "user", "content": "என்ன டாக்குமெண்ட்?"},
            {"role": "tool", "tool_call_id": "1", "content": "{}"},
        )
    )
    assert run.spoken == ["ஆதார் ", "கார்டு வேணும்."]  # piece by piece, not held


async def test_text_is_not_held_without_tools(settings, monkeypatch):
    llm = build_llm(settings, "prompt")  # hold_text_before_tools stays off
    run = _Streamed(llm, monkeypatch, text=["வணக்கம், ", "சொல்லுங்க."], tool_call=False)
    await llm._process_context(_context({"role": "user", "content": "ஹலோ"}))
    assert run.spoken == ["வணக்கம், ", "சொல்லுங்க."]


async def test_answer_prompt_is_used_after_a_tool_result(settings):
    """LLM 1 request keeps the deciding prompt; LLM 2 (after the tool result) gets the
    answering prompt; the topic note still lands on whichever prompt is active."""
    llm = build_llm(settings, "decide prompt")
    llm.answer_instruction = "answer prompt"
    convo = [{"role": "user", "content": "கேள்வி"}]

    def params_for(messages):
        return llm.build_chat_completion_params(
            {"messages": [{"role": "system", "content": "decide prompt"}, *messages]}
        )

    assert params_for(convo)["messages"][0]["content"] == "decide prompt"
    after_tool = params_for(
        [
            *convo,
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "{}"},
        ]
    )
    assert after_tool["messages"][0]["content"] == "answer prompt"
    assert after_tool["messages"][-1]["role"] == "tool"  # the rest is untouched
