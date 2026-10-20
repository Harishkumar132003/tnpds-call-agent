import pytest

from voice_agent.languages import (
    SUPPORTED_LANGUAGES,
    UnsupportedLanguageError,
    get_language,
    supported_language_codes,
)


def test_hindi_is_supported():
    lang = get_language("hi-IN")
    assert lang.english_name == "Hindi"
    assert lang.pipecat_language.value == "hi-IN"


def test_all_registered_codes_resolve():
    for code in supported_language_codes():
        assert get_language(code).code == code
    assert len(SUPPORTED_LANGUAGES) == len(supported_language_codes())


def test_unsupported_language_lists_supported_codes():
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        get_language("fr-FR")
    assert "fr-FR" in str(excinfo.value)
    assert "hi-IN" in str(excinfo.value)
