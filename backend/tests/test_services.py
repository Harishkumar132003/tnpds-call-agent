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
