"""Live integration tests against the real Sarvam API.

They are skipped unless ``SARVAM_API_KEY`` is set and are excluded from the
default ``pytest`` run. Run them with::

    SARVAM_API_KEY=... .venv/bin/python -m pytest -m integration -q
"""

from __future__ import annotations

import os

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import SleepFrame, run_test

from voice_agent.config import Settings
from voice_agent.services import build_llm, build_stt, build_tts

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("SARVAM_API_KEY"), reason="SARVAM_API_KEY not set"),
]

HINDI_GREETING = "नमस्ते, आप कैसे हैं?"
STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
FRAME_MS = 20


@pytest.fixture
def live_settings() -> Settings:
    return Settings(_env_file=None, agent_language="hi-IN")


async def synthesize(settings: Settings, text: str) -> bytes:
    tts = build_tts(settings)
    down, _ = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text), SleepFrame(4.0)],
        pipeline_params=PipelineParams(audio_out_sample_rate=TTS_SAMPLE_RATE),
    )
    audio_frames = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames, f"no TTS audio received; frames: {[type(f).__name__ for f in down]}"
    assert audio_frames[0].sample_rate == TTS_SAMPLE_RATE
    return b"".join(f.audio for f in audio_frames)


def _resample_24k_to_16k(pcm: bytes) -> bytes:
    import numpy as np
    import soxr

    samples = np.frombuffer(pcm, dtype=np.int16)
    return soxr.resample(samples, TTS_SAMPLE_RATE, STT_SAMPLE_RATE).astype(np.int16).tobytes()


def _chunk_audio(pcm: bytes, sample_rate: int) -> list[InputAudioRawFrame]:
    bytes_per_frame = sample_rate * 2 * FRAME_MS // 1000
    return [
        InputAudioRawFrame(pcm[i : i + bytes_per_frame], sample_rate, 1)
        for i in range(0, len(pcm), bytes_per_frame)
    ]


async def test_tts_produces_audio(live_settings):
    audio = await synthesize(live_settings, HINDI_GREETING)
    seconds = len(audio) / (TTS_SAMPLE_RATE * 2)
    assert 0.5 < seconds < 10, f"unexpected audio length {seconds:.2f}s"


async def test_tts_then_stt_round_trip(live_settings):
    """Speech synthesised by Sarvam TTS must be transcribed back by Sarvam STT."""
    audio_16k = _resample_24k_to_16k(await synthesize(live_settings, HINDI_GREETING))
    frames = [
        VADUserStartedSpeakingFrame(),
        *_chunk_audio(audio_16k, STT_SAMPLE_RATE),
        VADUserStoppedSpeakingFrame(),  # makes the STT service flush the utterance
        SleepFrame(5.0),
    ]
    stt = build_stt(live_settings)
    down, _ = await run_test(
        stt,
        frames_to_send=frames,
        pipeline_params=PipelineParams(audio_in_sample_rate=STT_SAMPLE_RATE),
    )
    transcripts = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert transcripts, f"no transcription; frames: {[type(f).__name__ for f in down]}"
    text = " ".join(t.text for t in transcripts)
    assert "नमस्ते" in text or "कैसे" in text, f"unexpected transcript: {text!r}"
    assert transcripts[0].language is not None


async def test_llm_replies_in_hindi(live_settings):
    llm = build_llm(
        live_settings,
        "You are a voice assistant. Reply in Hindi (Devanagari script) in one short sentence.",
    )
    context = LLMContext(messages=[{"role": "user", "content": "नमस्ते, आप कौन हैं?"}])
    down, _ = await run_test(
        llm,
        frames_to_send=[LLMContextFrame(context), SleepFrame(10.0)],
    )
    assert any(isinstance(f, LLMFullResponseEndFrame) for f in down)
    reply = "".join(f.text for f in down if isinstance(f, LLMTextFrame))
    assert reply.strip(), "empty LLM reply"
    assert any("ऀ" <= ch <= "ॿ" for ch in reply), f"not Devanagari: {reply!r}"
