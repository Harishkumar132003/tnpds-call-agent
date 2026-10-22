"""Sarvam model catalogue, one list per pipeline step.

The portal offers only what fits the step and the conversation language, and
the session validates portal choices against the same lists. Everything here
was checked against Sarvam's API on 2026-10-21: every chat model answered in
Tamil script and made a tool call, every bulbul:v3 voice synthesised Tamil,
bulbul:v2 rejected Tamil, and only saaras:v3/v4 work over the streaming
websocket. Latencies are one short Tamil reply, measured once; indicative only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voice_agent.languages import supported_language_codes

ALL_LANGUAGES = frozenset(supported_language_codes())


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    note: str
    languages: frozenset[str] = ALL_LANGUAGES
    recommended: bool = False

    def supports(self, language: str) -> bool:
        return language in self.languages


@dataclass(frozen=True)
class Speaker:
    id: str
    gender: str
    """``male``/``female``, or ``""`` when Sarvam does not say (bulbul:v4-flash personas)."""
    tier: int
    """Sarvam's production tiers by critical error rate: 1 best, 2 good, 3 unranked."""
    recommended_for: frozenset[str] = frozenset()
    """Languages Sarvam's docs name this voice for."""
    languages: frozenset[str] = ALL_LANGUAGES
    """Languages the voice can speak (v3 voices speak all; v4-flash voices are per language)."""
    style: str | None = None
    """Persona style baked into a v4-flash voice (``narration``, ``customer`` ...)."""

    def supports(self, language: str) -> bool:
        return language in self.languages


STT_MODELS: list[ModelOption] = [
    ModelOption("saaras:v4", "Saaras v4", "current, recommended: 23 languages", recommended=True),
    ModelOption("saaras:v3", "Saaras v3", "previous generation, still served"),
]

_V4_FLASH_LANGUAGE = {
    "hi": ("hi-IN",),
    "en": ("en-IN",),
    "enhi": ("en-IN", "hi-IN"),
    "bn": ("bn-IN",),
    "gu": ("gu-IN",),
    "kn": ("kn-IN",),
    "mr": ("mr-IN",),
    "pa": ("pa-IN",),
    "ta": ("ta-IN",),
    "te": ("te-IN",),
}
"""Language segment of a v4-flash voice id (``aravind_ta_ads``) to agent language codes.
Assamese voices exist but the agent does not support as-IN; no Malayalam or Odia voices."""


def _v4_flash_voices() -> list[Speaker]:
    data = json.loads((Path(__file__).parent / "data/bulbul_v4_flash_voices.json").read_text())
    voices = []
    for voice_id in data["voices"]:
        name, lang, style = voice_id.split("_", 2)
        codes = _V4_FLASH_LANGUAGE.get(lang)
        if codes is None:
            continue
        voices.append(Speaker(voice_id, "", 3, languages=frozenset(codes), style=style))
    return voices


V4_FLASH_VOICES = _v4_flash_voices()
V4_FLASH_LANGUAGES = frozenset(code for v in V4_FLASH_VOICES for code in v.languages)

TTS_MODELS: list[ModelOption] = [
    ModelOption(
        "bulbul:v3",
        "Bulbul v3",
        "37 voices, each speaks all 11 languages; Sarvam's per-language picks",
        recommended=True,
    ),
    ModelOption(
        "bulbul:v4-flash",
        "Bulbul v4 Flash",
        f"newest: {len(V4_FLASH_VOICES)} persona voices, each for one language and style",
        languages=V4_FLASH_LANGUAGES,
    ),
]

_M, _F = "male", "female"
_TIER1 = {"mani", "priya", "ishita", "varun"}
_TIER2 = {"roopa", "shubh", "pooja", "sunny", "ratan", "rehan", "ashutosh", "amit", "rahul"}
_RECOMMENDED: dict[str, tuple[str, ...]] = {
    "en-IN": ("ratan", "ishita"),
    "hi-IN": ("shubh", "ashutosh", "priya", "suhani"),
    "te-IN": ("shubh", "ratan", "neha", "priya"),
    "kn-IN": ("shubh", "ratan", "neha", "ishita"),
    "bn-IN": ("rehan", "roopa", "suhani"),
    "ta-IN": ("ratan", "rohan", "ishita", "ritu"),
    "od-IN": ("shubh", "ritu", "pooja"),
    "ml-IN": ("shubh", "pooja"),
    "mr-IN": ("ratan", "priya", "ritu"),
    "pa-IN": ("mani", "roopa", "suhani"),
    "gu-IN": ("ratan", "priya", "ritu"),
}
_VOICES: list[tuple[str, str]] = [
    ("shubh", _M), ("aditya", _M), ("ritu", _F), ("priya", _F), ("neha", _F), ("rahul", _M),
    ("pooja", _F), ("rohan", _M), ("simran", _F), ("kavya", _F), ("amit", _M), ("dev", _M),
    ("ishita", _F), ("shreya", _F), ("ratan", _M), ("varun", _M), ("manan", _M), ("sumit", _M),
    ("roopa", _F), ("kabir", _M), ("aayan", _M), ("ashutosh", _M), ("advait", _M), ("anand", _M),
    ("tanya", _F), ("tarun", _M), ("sunny", _M), ("mani", _M), ("gokul", _M), ("vijay", _M),
    ("shruti", _F), ("suhani", _F), ("mohit", _M), ("kavitha", _F), ("rehan", _M), ("soham", _M),
    ("rupali", _F),
]  # fmt: skip


def _speaker(name: str, gender: str) -> Speaker:
    tier = 1 if name in _TIER1 else 2 if name in _TIER2 else 3
    langs = frozenset(code for code, names in _RECOMMENDED.items() if name in names)
    return Speaker(name, gender, tier, langs)


TTS_SPEAKERS: dict[str, list[Speaker]] = {
    "bulbul:v3": [_speaker(n, g) for n, g in _VOICES],
    "bulbul:v4-flash": V4_FLASH_VOICES,
}

LLM_MODELS: list[ModelOption] = [
    ModelOption(
        "sarvam-105b", "Sarvam 105B", "Sarvam's own model; ~0.6 s to first reply", recommended=True
    ),
]
"""Sarvam's own chat models only. Sarvam's API also serves open models (GLM, Gemma,
DeepSeek); they answered in Tamil and called tools in probes but are left out by choice."""
_LLM_REASONING: dict[str, str] = {}
"""Models that reject ``reasoning_effort="none"``: the lowest value they accept."""

TRANSLATE_MODELS: list[ModelOption] = [
    ModelOption(
        "sarvam-translate:v1", "Sarvam Translate v1", "23 languages, formal", recommended=True
    ),
    ModelOption("mayura:v1", "Mayura v1", "11 languages, colloquial modes"),
]

EMBEDDING_MODELS: list[ModelOption] = [
    ModelOption(
        "text-embedding-3-small",
        "OpenAI text-embedding-3-small",
        "1536 dims, multilingual, lowest cost",
        recommended=True,
    ),
    ModelOption(
        "text-embedding-3-large",
        "OpenAI text-embedding-3-large",
        "3072 dims, higher recall, ~6x cost",
    ),
    ModelOption("text-embedding-ada-002", "OpenAI text-embedding-ada-002", "1536 dims, legacy"),
]
"""Embeds documents and questions for the knowledge base (OpenAI). Changing it means
re-embedding every document, which the portal does on save."""

STEPS: dict[str, list[ModelOption]] = {
    "stt": STT_MODELS,
    "llm": LLM_MODELS,
    "tts": TTS_MODELS,
    "translate": TRANSLATE_MODELS,
    "embedding": EMBEDDING_MODELS,
}


def find(step: str, model_id: str) -> ModelOption | None:
    return next((m for m in STEPS[step] if m.id == model_id), None)


def speakers_for(model_id: str, language: str | None = None) -> list[str]:
    """Voice ids for a model, optionally only those that speak ``language``."""
    return [
        s.id for s in TTS_SPEAKERS.get(model_id, []) if language is None or s.supports(language)
    ]


def default_speaker(model_id: str, language: str) -> str | None:
    """A sensible voice for a model and language: Sarvam's pick, else the first that fits."""
    voices = [s for s in TTS_SPEAKERS.get(model_id, []) if s.supports(language)]
    ids = {s.id for s in voices}
    preferred = [name for name in _RECOMMENDED.get(language, ()) if name in ids]  # docs order
    if preferred:
        return preferred[0]
    return voices[0].id if voices else None


def reasoning_effort_for(model_id: str, configured: str | None) -> str | None:
    """The reasoning setting to send: the configured one unless the model rejects it."""
    minimum = _LLM_REASONING.get(model_id)
    if minimum and configured in (None, "none"):
        return minimum
    return configured


def validate_selection(
    language: str,
    *,
    stt_model: str | None = None,
    llm_model: str | None = None,
    tts_model: str | None = None,
    tts_speaker: str | None = None,
    translate_model: str | None = None,
    embedding_model: str | None = None,
) -> None:
    """Raise ``ValueError`` naming the first choice that does not fit the step or language."""
    for step, chosen in (
        ("stt", stt_model),
        ("llm", llm_model),
        ("tts", tts_model),
        ("translate", translate_model),
        ("embedding", embedding_model),
    ):
        if chosen is None:
            continue
        option = find(step, chosen)
        if option is None:
            allowed = ", ".join(m.id for m in STEPS[step])
            raise ValueError(f"{step} model {chosen!r} is not available; choose one of {allowed}")
        if not option.supports(language):
            raise ValueError(f"{step} model {chosen!r} does not support {language}")
    if tts_speaker is not None:
        model = tts_model or TTS_MODELS[0].id
        if tts_speaker not in speakers_for(model):
            raise ValueError(f"voice {tts_speaker!r} does not exist for {model}")
        if tts_speaker not in speakers_for(model, language):
            raise ValueError(f"voice {tts_speaker!r} does not speak {language}")


def catalogue(language: str) -> dict:
    """JSON-ready lists for the portal, limited to what supports ``language``."""

    def options(step: str) -> list[dict]:
        return [
            {"id": m.id, "label": m.label, "note": m.note, "recommended": m.recommended}
            for m in STEPS[step]
            if m.supports(language)
        ]

    voices = {
        model: [
            {
                "id": s.id,
                "gender": s.gender,
                "tier": s.tier,
                "style": s.style,
                "recommended": language in s.recommended_for,
            }
            for s in speakers
            if s.supports(language)
        ]
        for model, speakers in TTS_SPEAKERS.items()
    }
    return {
        "language": language,
        "stt": options("stt"),
        "llm": options("llm"),
        "tts": options("tts"),
        "translate": options("translate"),
        "embedding": options("embedding"),
        "voices": voices,
    }
