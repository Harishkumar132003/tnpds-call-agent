"""Headless WebRTC client for exercising the bot without a browser.

It performs the same signalling as the web client (``POST /api/offer``),
sends microphone audio from a WAV file (or silence), records everything the
bot sends back to a WAV file, and prints connection/latency figures.

Examples::

    # 20 s session with a silent microphone; saves the bot's greeting
    python scripts/webrtc_probe.py --duration 20 --out greeting.wav

    # play a recorded Hindi question into the bot and capture the answer
    python scripts/webrtc_probe.py --wav question_16k.wav --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import time
import wave
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from aiortc.mediastreams import AudioStreamTrack, MediaStreamTrack

SILENCE_RMS_THRESHOLD = 50  # int16 RMS below which a frame is treated as silence


@dataclass
class ProbeStats:
    started_at: float = field(default_factory=time.monotonic)
    connected_at: float | None = None
    first_audio_at: float | None = None
    first_speech_at: float | None = None
    frames: int = 0
    speech_frames: int = 0
    sample_rate: int | None = None
    channels: int | None = None
    states: list[tuple[float, str]] = field(default_factory=list)

    def mark(self, label: str) -> None:
        self.states.append((time.monotonic() - self.started_at, label))


async def _record_remote_audio(track: MediaStreamTrack, out_path: str, stats: ProbeStats) -> None:
    wav: wave.Wave_write | None = None
    try:
        while True:
            frame = await track.recv()
            pcm = frame.to_ndarray()  # shape (channels, samples) or (1, channels*samples)
            if wav is None:
                stats.sample_rate = frame.sample_rate
                stats.channels = len(frame.layout.channels)
                wav = wave.open(out_path, "wb")  # noqa: SIM115 - closed in finally
                wav.setnchannels(stats.channels)
                wav.setsampwidth(2)
                wav.setframerate(frame.sample_rate)
                stats.first_audio_at = time.monotonic()
            wav.writeframes(pcm.astype(np.int16).tobytes())
            stats.frames += 1
            rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
            if rms > SILENCE_RMS_THRESHOLD:
                stats.speech_frames += 1
                if stats.first_speech_at is None:
                    stats.first_speech_at = time.monotonic()
    except Exception as exc:  # MediaStreamError when the track ends; anything else is worth seeing
        if type(exc).__name__ != "MediaStreamError":
            print(f"remote audio reader stopped: {type(exc).__name__}: {exc}")
    finally:
        if wav is not None:
            wav.close()


async def run_probe(server: str, wav_in: str | None, out_path: str, duration: float) -> ProbeStats:
    stats = ProbeStats()
    pc = RTCPeerConnection()
    recorder_task: asyncio.Task | None = None

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        stats.mark(f"connection:{pc.connectionState}")
        if pc.connectionState == "connected" and stats.connected_at is None:
            stats.connected_at = time.monotonic()

    @pc.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        nonlocal recorder_task
        stats.mark(f"remote-track:{track.kind}")
        if track.kind == "audio":
            recorder_task = asyncio.create_task(_record_remote_audio(track, out_path, stats))

    mic: MediaStreamTrack = MediaPlayer(wav_in).audio if wav_in else AudioStreamTrack()
    pc.addTrack(mic)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)  # aiortc finishes ICE gathering here
    async with (
        aiohttp.ClientSession() as http,
        http.post(
            f"{server}/api/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        ) as resp,
    ):
        body = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"/api/offer failed: {resp.status} {body}")
    await pc.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
    stats.mark(f"answer-received pc_id={body.get('pc_id')}")

    deadline = time.monotonic() + duration
    while time.monotonic() < deadline and pc.connectionState not in {"failed", "closed"}:
        await asyncio.sleep(0.25)

    stats.mark("probe-finished")
    await pc.close()
    if recorder_task:
        recorder_task.cancel()
        await asyncio.gather(recorder_task, return_exceptions=True)
    return stats


def _print_report(stats: ProbeStats, out_path: str) -> None:
    def rel(t: float | None) -> str:
        return "n/a" if t is None else f"{t - stats.started_at:6.2f}s"

    print("--- WebRTC probe report ---")
    for t, label in stats.states:
        print(f"  {t:6.2f}s  {label}")
    print(f"connected at          : {rel(stats.connected_at)}")
    print(f"first remote audio at : {rel(stats.first_audio_at)}")
    print(f"first bot speech at   : {rel(stats.first_speech_at)}")
    print(f"audio frames received : {stats.frames} ({stats.speech_frames} non-silent)")
    if stats.sample_rate:
        print(f"remote audio format   : {stats.sample_rate} Hz, {stats.channels} ch -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server", default="http://localhost:7860")
    parser.add_argument("--wav", help="WAV file to play as the microphone (default: silence)")
    parser.add_argument("--out", default="bot_output.wav", help="where to save the bot's audio")
    parser.add_argument("--duration", type=float, default=15.0, help="seconds to stay connected")
    args = parser.parse_args()
    stats = asyncio.run(run_probe(args.server, args.wav, args.out, args.duration))
    _print_report(stats, args.out)
    if stats.connected_at is None:
        raise SystemExit("probe did not reach the connected state")


if __name__ == "__main__":
    main()
