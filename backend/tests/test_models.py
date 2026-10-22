"""The Sarvam model catalogue: filtering by step and language, validation, overrides."""

from __future__ import annotations

import pytest

from voice_agent import models as catalogue
from voice_agent.services import build_llm, build_tts
from voice_agent.session import apply_model_overrides
from voice_agent.tracking.config_store import AgentConfig


def test_catalogue_offers_only_models_for_the_step_and_language():
    c = catalogue.catalogue("ta-IN")
    assert [m["id"] for m in c["stt"]] == ["saaras:v4", "saaras:v3"]
    assert [m["id"] for m in c["tts"]] == ["bulbul:v3", "bulbul:v4-flash"]  # v2 rejects Tamil
    assert [m["id"] for m in c["llm"]] == ["sarvam-105b"]  # Sarvam's own models only
    assert [m["id"] for m in c["embedding"]] == [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]
    assert {m["id"] for m in c["translate"]} == {"sarvam-translate:v1", "mayura:v1"}
    voices = c["voices"]["bulbul:v3"]
    assert len(voices) == 37
    assert {v["id"] for v in voices if v["recommended"]} == {"ratan", "rohan", "ishita", "ritu"}
    ishita = next(v for v in voices if v["id"] == "ishita")
    assert ishita["gender"] == "female" and ishita["tier"] == 1


def test_validate_selection_names_the_bad_choice():
    catalogue.validate_selection(
        "ta-IN",
        llm_model="sarvam-105b",
        tts_speaker="anand",
        embedding_model="text-embedding-3-large",
    )
    with pytest.raises(ValueError, match="llm model 'gemma4'"):
        catalogue.validate_selection("ta-IN", llm_model="gemma4")
    with pytest.raises(ValueError, match="embedding model 'bge'"):
        catalogue.validate_selection("ta-IN", embedding_model="bge")
    with pytest.raises(ValueError, match="stt model 'whisper'"):
        catalogue.validate_selection("ta-IN", stt_model="whisper")
    with pytest.raises(ValueError, match="voice 'anushka'"):
        catalogue.validate_selection("ta-IN", tts_speaker="anushka")


def test_reasoning_effort_passes_through_for_sarvam_models():
    assert catalogue.reasoning_effort_for("sarvam-105b", "none") == "none"
    assert catalogue.reasoning_effort_for("sarvam-105b", "high") == "high"


def test_apply_model_overrides(settings):
    config = AgentConfig(
        stt_model="saaras:v3",
        tts_speaker="anand",
        translate_model="mayura:v1",
        embedding_model="text-embedding-3-large",
    )
    out = apply_model_overrides(settings, config)
    assert (out.sarvam_stt_model, out.sarvam_tts_speaker, out.translate_model) == (
        "saaras:v3",
        "anand",
        "mayura:v1",
    )
    assert (out.embedding_model, out.embedding_dimensions) == ("text-embedding-3-large", None)
    assert out.llm_model == settings.llm_model  # untouched
    # a choice that no longer exists leaves the environment settings alone
    assert apply_model_overrides(settings, AgentConfig(tts_speaker="anushka")) is settings
    assert apply_model_overrides(settings, AgentConfig()) is settings


def test_services_accept_catalogue_choices(settings):
    s = settings.model_copy(update={"sarvam_tts_speaker": "anand"})
    assert build_tts(s)._settings.voice == "anand"
    assert build_llm(s, "hi")._settings.model == "sarvam-105b"


def test_v4_flash_voices_are_per_language():
    ta = catalogue.catalogue("ta-IN")
    assert [m["id"] for m in ta["tts"]] == ["bulbul:v3", "bulbul:v4-flash"]
    v4 = ta["voices"]["bulbul:v4-flash"]
    assert {v["id"] for v in v4} == {
        "aravind_ta_ads",
        "aravind_ta_suspense",
        "gokul_ta_narration",
        "vijay_ta_narration",
    }
    assert v4[0]["style"] and v4[0]["gender"] == ""
    assert len(catalogue.catalogue("hi-IN")["voices"]["bulbul:v4-flash"]) > 100  # hi + enhi
    # no Malayalam voices: the model itself is not offered
    assert [m["id"] for m in catalogue.catalogue("ml-IN")["tts"]] == ["bulbul:v3"]
    catalogue.validate_selection(
        "ta-IN", tts_model="bulbul:v4-flash", tts_speaker="vijay_ta_narration"
    )
    with pytest.raises(ValueError, match="does not speak ta-IN"):
        catalogue.validate_selection(
            "ta-IN", tts_model="bulbul:v4-flash", tts_speaker="shubh_enhi_ads"
        )
    assert catalogue.default_speaker("bulbul:v3", "ta-IN") == "ratan"
    assert catalogue.default_speaker("bulbul:v4-flash", "ta-IN") == "aravind_ta_ads"


def test_build_tts_accepts_v4_flash(settings):
    s = settings.model_copy(
        update={"sarvam_tts_model": "bulbul:v4-flash", "sarvam_tts_speaker": "aravind_ta_ads"}
    )
    tts = build_tts(s)
    assert tts._settings.model == "bulbul:v4-flash" and tts._settings.voice == "aravind_ta_ads"


def test_every_sarvam_model_has_a_price_and_openai_none():
    embedding = {m.id for m in catalogue.EMBEDDING_MODELS}
    missing = [m.id for m in catalogue.all_models() if m.price is None and m.id not in embedding]
    assert missing == []
    assert all(m.price is None for m in catalogue.EMBEDDING_MODELS)


def test_price_definitions_cover_the_sarvam_models():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "langfuse_models", Path(__file__).parent.parent / "scripts/langfuse_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defs = {d["modelName"]: d for d in module.definitions()}
    assert set(defs) == {m.id for m in catalogue.all_models() if m.price is not None}
    assert "text-embedding-3-small" not in defs
    assert defs["bulbul:v4-flash"]["outputPrice"] == 0.002
    assert defs["sarvam-translate:v1"]["unit"] == "CHARACTERS"
    assert defs["saaras:v3"]["inputPrice"] == defs["saaras:v4"]["inputPrice"]
