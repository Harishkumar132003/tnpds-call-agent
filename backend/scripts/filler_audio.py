"""Record the knowledge-search filler lines with the agent's current voice.

Writes one WAV per line to ``voice_agent/assets/fillers/`` plus ``manifest.json``
(text, model, speaker, pace, dictionary) so a later run can tell whether the
clips still match the configured voice. Run from ``backend/``:

    .venv/bin/python scripts/filler_audio.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent.config import Settings  # noqa: E402
from voice_agent.knowledge.tool import SEARCHING_LINES  # noqa: E402
from voice_agent.tracking.config_store import load_active_config  # noqa: E402
from voice_agent.tracking.db import create_pool  # noqa: E402

TTS_URL = "https://api.sarvam.ai/text-to-speech"
SAMPLE_RATE = 24000  # the web transport's rate; telephony resamples to 8 kHz on output
OUT_DIR = Path(__file__).resolve().parent.parent / "voice_agent" / "assets" / "fillers"


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    settings = Settings()
    pool = await create_pool(os.environ["DATABASE_URL"])
    try:
        config = await load_active_config(pool)
    finally:
        await pool.close()

    language = config.language or settings.agent_language
    model = config.tts_model or settings.sarvam_tts_model
    speaker = config.tts_speaker or settings.sarvam_tts_speaker
    pace = settings.sarvam_tts_pace
    dict_id = config.tts_dict_id
    lines = SEARCHING_LINES.get(language, SEARCHING_LINES["en-IN"])
    if isinstance(lines, str):
        lines = (lines,)
    print(f"voice: {model} / {speaker} / pace {pace} / dict {dict_id or '-'} / {language}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = []
    async with httpx.AsyncClient(timeout=60) as client:
        for n, text in enumerate(lines, start=1):
            body = {
                "text": text,
                "target_language_code": language,
                "model": model,
                "speaker": speaker,
                "pace": pace,
                "speech_sample_rate": SAMPLE_RATE,
                "output_audio_codec": "wav",
            }
            if dict_id:
                body["dict_id"] = dict_id
            response = await client.post(
                TTS_URL,
                json=body,
                headers={"api-subscription-key": settings.sarvam_api_key.get_secret_value()},
            )
            if response.status_code != 200:
                raise SystemExit(f"line {n}: HTTP {response.status_code} {response.text[:300]}")
            audio = base64.b64decode(response.json()["audios"][0])
            path = OUT_DIR / f"{language}-{n}.wav"
            path.write_bytes(audio)
            with wave.open(str(path)) as wav:
                secs = wav.getnframes() / wav.getframerate()
                info = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
            print(f"  {path.name}: {secs:.2f}s rate/ch/width={info}  {text}")
            clips.append({"file": path.name, "text": text, "seconds": round(secs, 2)})

    manifest = {
        "language": language,
        "model": model,
        "speaker": speaker,
        "pace": pace,
        "dict_id": dict_id,
        "sample_rate": SAMPLE_RATE,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "clips": clips,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (OUT_DIR / "manifest.json").write_text(text)
    print(f"wrote {len(clips)} clips + manifest to {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
