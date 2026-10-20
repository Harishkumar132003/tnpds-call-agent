"""Registry of conversation languages supported by both Sarvam STT and TTS.

Sarvam uses BCP-47 style codes such as ``hi-IN``. Pipecat's Sarvam services
accept ``pipecat.transcriptions.language.Language`` values and translate them
to Sarvam codes; this module keeps the two in sync and exposes a single place
to validate user configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.transcriptions.language import Language


class UnsupportedLanguageError(ValueError):
    """Raised when a language code is not supported by the agent."""


@dataclass(frozen=True)
class SupportedLanguage:
    """A language the agent can both listen to and speak."""

    code: str
    """Sarvam language code, e.g. ``hi-IN``."""
    english_name: str
    native_name: str
    pipecat_language: Language


_LANGUAGES: tuple[SupportedLanguage, ...] = (
    SupportedLanguage("hi-IN", "Hindi", "हिन्दी", Language.HI_IN),
    SupportedLanguage("en-IN", "English (India)", "English", Language.EN_IN),
    SupportedLanguage("bn-IN", "Bengali", "বাংলা", Language.BN_IN),
    SupportedLanguage("gu-IN", "Gujarati", "ગુજરાતી", Language.GU_IN),
    SupportedLanguage("kn-IN", "Kannada", "ಕನ್ನಡ", Language.KN_IN),
    SupportedLanguage("ml-IN", "Malayalam", "മലയാളം", Language.ML_IN),
    SupportedLanguage("mr-IN", "Marathi", "मराठी", Language.MR_IN),
    SupportedLanguage("od-IN", "Odia", "ଓଡ଼ିଆ", Language.OR_IN),
    SupportedLanguage("pa-IN", "Punjabi", "ਪੰਜਾਬੀ", Language.PA_IN),
    SupportedLanguage("ta-IN", "Tamil", "தமிழ்", Language.TA_IN),
    SupportedLanguage("te-IN", "Telugu", "తెలుగు", Language.TE_IN),
)

SUPPORTED_LANGUAGES: dict[str, SupportedLanguage] = {lang.code: lang for lang in _LANGUAGES}


def supported_language_codes() -> list[str]:
    """Return the supported Sarvam language codes in display order."""
    return [lang.code for lang in _LANGUAGES]


def get_language(code: str) -> SupportedLanguage:
    """Look up a supported language by Sarvam code.

    Raises:
        UnsupportedLanguageError: if ``code`` is not supported.
    """
    try:
        return SUPPORTED_LANGUAGES[code]
    except KeyError:
        supported = ", ".join(supported_language_codes())
        raise UnsupportedLanguageError(
            f"Unsupported language {code!r}. Supported languages: {supported}"
        ) from None
