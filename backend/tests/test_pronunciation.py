"""Pronunciation dictionary parsing, file format, and TTS config wiring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from voice_agent.pronunciation import (
    PronunciationEntry,
    dictionary_file,
    entries_to_text,
    parse_entries,
)
from voice_agent.services import build_tts


def test_parse_entries_accepts_word_equals_spoken_lines():
    text = (
        "# acronyms\nTNPDS = டி என் பி டி எஸ்\n\nOTP=ஓ டி பி\ntnpds.gov.in = tnpds dot gov dot in\n"
    )
    entries = parse_entries(text)
    assert entries == [
        PronunciationEntry("TNPDS", "டி என் பி டி எஸ்"),
        PronunciationEntry("OTP", "ஓ டி பி"),
        PronunciationEntry("tnpds.gov.in", "tnpds dot gov dot in"),
    ]
    assert (
        entries_to_text([{"word": e.word, "spoken": e.spoken} for e in entries]).splitlines()[0]
        == "TNPDS = டி என் பி டி எஸ்"
    )


def test_parse_entries_rejects_bad_lines_and_too_many():
    with pytest.raises(ValueError, match="line 2"):
        parse_entries("OK = fine\nnot a pair")
    with pytest.raises(ValueError, match="line 1"):
        parse_entries("word =")
    with pytest.raises(ValueError, match="at most 100"):
        parse_entries("\n".join(f"w{i} = s{i}" for i in range(101)))
    assert parse_entries("TNPDS = a\nTNPDS = b") == [PronunciationEntry("TNPDS", "b")]  # last wins


def test_dictionary_file_matches_sarvam_format():
    data = json.loads(dictionary_file("ta-IN", [PronunciationEntry("OTP", "ஓ டி பி")]))
    assert data == {"pronunciations": {"ta-IN": {"OTP": "ஓ டி பி"}}}


async def test_tts_config_includes_dict_id(settings):
    tts = build_tts(settings, dict_id="p_abc123")
    tts._websocket = AsyncMock()
    tts._speech_sample_rate = 24000
    tts._output_audio_codec = "linear16"
    tts._output_audio_bitrate = "128k"
    await tts._send_config()
    sent = json.loads(tts._websocket.send.await_args.args[0])
    assert sent["type"] == "config" and sent["data"]["dict_id"] == "p_abc123"
    assert sent["data"]["speaker"] == "shubh" and sent["data"]["model"] == "bulbul:v3"

    plain = build_tts(settings)
    assert "dict_id" not in plain._config_data()
