import json
import math
import struct
import wave

import pytest
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

from voice_agent.fillers import ASSETS_DIR, FillerClips, Voice, load_filler_clips
from voice_agent.knowledge.tool import SEARCHING_LINES

VOICE = Voice(model="bulbul:v4-flash", speaker="aravind_ta_ads", pace=1.0, dict_id=None)


def _tone(path, *, seconds: float, rate: int = 24000) -> None:
    frames = int(seconds * rate)
    samples = (int(12000 * math.sin(2 * math.pi * 440 * n / rate)) for n in range(frames))
    pcm = b"".join(struct.pack("<h", sample) for sample in samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)


@pytest.fixture
def clip_dir(tmp_path):
    _tone(tmp_path / "a.wav", seconds=1.0)
    _tone(tmp_path / "b.wav", seconds=0.5)
    manifest = {
        "language": "ta-IN",
        "model": VOICE.model,
        "speaker": VOICE.speaker,
        "pace": 1.0,
        "dict_id": None,
        "sample_rate": 24000,
        "clips": [{"file": "a.wav", "text": "line a"}, {"file": "b.wav", "text": "line b"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_load_reads_manifest_and_wavs(clip_dir):
    clips = FillerClips.load("ta-IN", clip_dir)
    assert clips is not None
    assert len(clips) == 2 and clips.texts == ("line a", "line b")
    assert clips.voice == VOICE
    assert FillerClips.load("hi-IN", clip_dir) is None  # other language: nothing recorded
    assert FillerClips.load("ta-IN", clip_dir / "missing") is None


def test_voice_must_match_the_manifest(clip_dir):
    clips = FillerClips.load("ta-IN", clip_dir)
    assert clips.matches(VOICE)
    assert not clips.matches(Voice("bulbul:v4-flash", "ishita", 1.0, None))
    assert not clips.matches(Voice("bulbul:v4-flash", "aravind_ta_ads", 1.2, None))
    assert not clips.matches(Voice("bulbul:v4-flash", "aravind_ta_ads", 1.0, "dict-1"))


async def test_frames_look_like_one_tts_utterance(clip_dir):
    clips = FillerClips.load("ta-IN", clip_dir)
    frames = await clips.frames("line a", 24000)
    assert isinstance(frames[0], TTSStartedFrame) and frames[0].append_to_context is False
    assert isinstance(frames[-1], TTSStoppedFrame)
    audio = frames[1:-1]
    assert all(isinstance(f, TTSAudioRawFrame) for f in audio)
    assert {f.sample_rate for f in audio} == {24000} and {f.num_channels for f in audio} == {1}
    assert sum(len(f.audio) for f in audio) == 24000 * 2  # the whole second, nothing dropped
    assert len(audio) == 5  # 200 ms chunks
    assert await clips.frames("not recorded", 24000) is None


async def test_telephony_rate_is_resampled_once_and_reused(clip_dir):
    clips = FillerClips.load("ta-IN", clip_dir)
    first = await clips.frames("line b", 8000)
    audio = first[1:-1]
    assert {f.sample_rate for f in audio} == {8000}
    assert sum(len(f.audio) for f in audio) == pytest.approx(8000 * 2 * 0.5, rel=0.01)
    again = await clips.frames("line b", 8000)
    assert b"".join(f.audio for f in again[1:-1]) == b"".join(f.audio for f in audio)
    assert set(clips._at_rate) == {8000}  # 24 kHz never needed, never converted


def test_load_filler_clips_falls_back_when_the_voice_changed(clip_dir, monkeypatch):
    import voice_agent.fillers as fillers

    monkeypatch.setattr(fillers, "_loaded", {})
    assert load_filler_clips("ta-IN", VOICE, clip_dir) is not None
    assert load_filler_clips("ta-IN", Voice("bulbul:v3", "ishita", 1.0, None), clip_dir) is None
    assert load_filler_clips("hi-IN", VOICE, clip_dir) is None


def test_recorded_assets_cover_every_tamil_line():
    clips = FillerClips.load("ta-IN", ASSETS_DIR)
    assert clips is not None
    assert set(clips.texts) == set(SEARCHING_LINES["ta-IN"])
    assert clips.matches(VOICE)  # the voice the agent is configured with today
