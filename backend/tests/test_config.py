import pytest
from pydantic import ValidationError

from voice_agent.config import ConfigurationError, LLMProvider, Settings, load_settings


def test_defaults(settings):
    assert settings.agent_language == "hi-IN"
    assert settings.sarvam_stt_model == "saaras:v4"
    assert settings.sarvam_tts_model == "bulbul:v3"
    assert settings.llm_provider is LLMProvider.SARVAM
    assert settings.sarvam_tts_speaker is None
    assert settings.sarvam_api_key.get_secret_value() != ""


def test_secret_is_not_printed(settings):
    assert "sk_test" not in repr(settings)


def test_unsupported_language_is_rejected(make_settings):
    with pytest.raises(ValidationError) as excinfo:
        make_settings(agent_language="fr-FR")
    assert "Supported languages" in str(excinfo.value)


def test_openai_provider_requires_key(make_settings):
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        make_settings(llm_provider="openai", llm_model="gpt-4.1-mini")


def test_openai_provider_with_key(make_settings):
    s = make_settings(llm_provider="openai", llm_model="gpt-4.1-mini", openai_api_key="k")
    assert s.llm_provider is LLMProvider.OPENAI


def test_blank_optional_values_become_none(make_settings):
    s = make_settings(sarvam_tts_speaker="  ", openai_base_url="", agent_system_prompt="")
    assert s.sarvam_tts_speaker is None
    assert s.openai_base_url is None
    assert s.agent_system_prompt is None


def test_pace_must_be_positive(make_settings):
    with pytest.raises(ValidationError):
        make_settings(sarvam_tts_pace=0)
    assert make_settings(sarvam_tts_pace=2.5).sarvam_tts_pace == 2.5  # model range checked later


def test_log_level_normalised(make_settings):
    assert make_settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        make_settings(log_level="loud")


def test_load_settings_reports_missing_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="sarvam_api_key"):
        load_settings()


def test_load_settings_reads_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SARVAM_API_KEY", "abc")
    monkeypatch.setenv("AGENT_LANGUAGE", "ta-IN")
    assert load_settings().agent_language == "ta-IN"


def test_settings_class_reads_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    (tmp_path / ".env").write_text("SARVAM_API_KEY=from-file\nAGENT_NAME=Meera\n")
    s = Settings()
    assert s.sarvam_api_key.get_secret_value() == "from-file"
    assert s.agent_name == "Meera"


def test_invalid_stt_mode_rejected(make_settings):
    with pytest.raises(ValidationError):
        make_settings(sarvam_stt_mode="shout")


def test_idle_timeout_must_be_positive(make_settings):
    with pytest.raises(ValidationError):
        make_settings(session_idle_timeout_secs=0)


def test_runner_args_from_settings(make_settings):
    from bot import runner_args_from_settings

    s = make_settings(public_host="https://callagent.wizzgeeks.com/", bot_transport="plivo")
    assert runner_args_from_settings([], s) == ["--proxy", "callagent.wizzgeeks.com", "-t", "plivo"]
    # explicit flags win
    assert runner_args_from_settings(["-t", "webrtc", "--proxy", "x"], s) == [
        "-t",
        "webrtc",
        "--proxy",
        "x",
    ]
    # nothing configured: argv untouched
    assert runner_args_from_settings(["--port", "7870"], make_settings()) == ["--port", "7870"]


def test_langfuse_configured_needs_all_three(make_settings):
    from voice_agent.tracking.langfuse_export import langfuse_configured

    full = dict(
        langfuse_host="http://localhost:3100", langfuse_public_key="pk", langfuse_secret_key="sk"
    )
    assert not langfuse_configured(make_settings())
    assert not langfuse_configured(make_settings(**full))  # off by default
    assert not langfuse_configured(
        make_settings(langfuse_enabled=True, langfuse_host="http://x", langfuse_public_key="pk")
    )
    assert langfuse_configured(make_settings(langfuse_enabled=True, **full))
    assert not langfuse_configured(make_settings(langfuse_enabled=False, **full))
