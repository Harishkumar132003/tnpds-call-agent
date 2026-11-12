"""Pre-recorded knowledge-search filler lines, played instead of synthesising them.

``scripts/filler_audio.py`` records every ``SEARCHING_LINES`` entry for a language
with the agent's voice into ``assets/fillers/`` and writes a manifest naming that
voice. A session plays a clip only when the manifest's voice is the one it
speaks with; otherwise the line goes through TTS as before, so a voice change
never makes the agent switch voices mid-call.

Clips are stored at one sample rate and resampled once per output rate (8 kHz
for telephony, 24 kHz for the web) the first time a session at that rate needs
them; after that every playback reuses the converted audio and the transport's
own resampler has nothing to do.
"""

from __future__ import annotations

import asyncio
import json
import wave
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pipecat.audio.utils import create_file_resampler
from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "fillers"
CHUNK_MS = 200
"""Clip audio is pushed in chunks this long so a barge-in drops at most one."""


@dataclass(frozen=True)
class Voice:
    """The TTS voice a session speaks with, as the manifest records it."""

    model: str
    speaker: str | None
    pace: float
    dict_id: str | None


@dataclass(frozen=True)
class Clip:
    text: str
    audio: bytes
    """16-bit mono PCM."""
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.audio) / (self.sample_rate * 2)


class FillerClips:
    """The recorded lines of one language, all spoken by ``voice``."""

    def __init__(self, language: str, voice: Voice, clips: list[Clip]) -> None:
        self.language = language
        self.voice = voice
        self._clips = {clip.text: clip for clip in clips}
        self._at_rate: dict[int, dict[str, Clip]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, language: str, directory: Path = ASSETS_DIR) -> FillerClips | None:
        """Read the manifest and WAVs for ``language``; None when there are none."""
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("language") != language:
            return None
        clips = []
        for entry in manifest.get("clips", []):
            path = directory / entry["file"]
            with wave.open(str(path)) as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError(f"{path.name}: filler clips must be 16-bit mono PCM")
                clips.append(
                    Clip(
                        text=entry["text"],
                        audio=wav.readframes(wav.getnframes()),
                        sample_rate=wav.getframerate(),
                    )
                )
        voice = Voice(
            model=manifest["model"],
            speaker=manifest.get("speaker"),
            pace=float(manifest.get("pace", 1.0)),
            dict_id=manifest.get("dict_id"),
        )
        return cls(language, voice, clips)

    def __len__(self) -> int:
        return len(self._clips)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(self._clips)

    def matches(self, voice: Voice) -> bool:
        return self.voice == voice

    async def frames(self, text: str, sample_rate: int) -> list[Frame] | None:
        """The frame sequence that plays ``text`` at ``sample_rate``; None if not recorded.

        Shaped like one TTS utterance (started, audio, stopped) so the transport
        reports bot speaking exactly as it does for synthesised speech, and
        with ``append_to_context=False`` so the aggregator ignores it, as it
        does for the spoken filler today.
        """
        clip = (await self._prepared(sample_rate)).get(text)
        if clip is None:
            return None
        chunk = sample_rate * 2 * CHUNK_MS // 1000
        frames: list[Frame] = [TTSStartedFrame(append_to_context=False)]
        for start in range(0, len(clip.audio), chunk):
            frames.append(
                TTSAudioRawFrame(
                    audio=clip.audio[start : start + chunk],
                    sample_rate=sample_rate,
                    num_channels=1,
                )
            )
        frames.append(TTSStoppedFrame())
        return frames

    async def _prepared(self, sample_rate: int) -> dict[str, Clip]:
        if sample_rate in self._at_rate:
            return self._at_rate[sample_rate]
        async with self._lock:
            if sample_rate not in self._at_rate:
                resampler = create_file_resampler()
                converted = {}
                for text, clip in self._clips.items():
                    audio = clip.audio
                    if clip.sample_rate != sample_rate:
                        audio = await resampler.resample(audio, clip.sample_rate, sample_rate)
                    converted[text] = Clip(text=text, audio=audio, sample_rate=sample_rate)
                self._at_rate[sample_rate] = converted
        return self._at_rate[sample_rate]


_loaded: dict[str, FillerClips | None] = {}


def load_filler_clips(
    language: str, voice: Voice, directory: Path = ASSETS_DIR
) -> FillerClips | None:
    """The clips to play for ``language``, or None when none are recorded.

    Read from disk once per process. The recordings are played whatever voice the
    session speaks with; when the manifest's voice differs from ``voice`` a warning
    says so, so the owner knows to re-record (scripts/filler_audio.py).
    """
    if language not in _loaded:
        try:
            _loaded[language] = FillerClips.load(language, directory)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning(f"Filler clips for {language} could not be loaded: {exc}")
            _loaded[language] = None
    clips = _loaded[language]
    if clips is None:
        logger.info(f"Filler clips: none recorded for {language}; filler lines go through TTS")
        return None
    if not clips.matches(voice):
        logger.warning(
            f"Filler clips: recorded with {clips.voice.model}/{clips.voice.speaker} at pace "
            f"{clips.voice.pace} but the agent speaks with {voice.model}/{voice.speaker} at pace "
            f"{voice.pace}; playing them anyway (re-run scripts/filler_audio.py to match)"
        )
    logger.info(
        f"Filler clips: {len(clips)} recorded lines ({voice.model}/{voice.speaker}) for {language}"
    )
    return clips
