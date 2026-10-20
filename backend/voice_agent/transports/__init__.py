"""Transport-specific configuration. Core agent logic never imports from here.

The Pipecat runner picks the transport from its CLI flag / request and calls
``create_transport(runner_args, TRANSPORT_PARAMS)``; the keys below are the
names it understands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.transports.base_transport import TransportParams

from voice_agent.transports.plivo import plivo_transport_params
from voice_agent.transports.webrtc import webrtc_transport_params

TRANSPORT_PARAMS: dict[str, Callable[[], TransportParams]] = {
    "webrtc": webrtc_transport_params,
    "plivo": plivo_transport_params,
}


@dataclass(frozen=True)
class AudioRates:
    """Sample rates the pipeline runs at (services are resampled to match)."""

    input_hz: int
    output_hz: int


WEB_AUDIO_RATES = AudioRates(input_hz=16000, output_hz=24000)
TELEPHONY_AUDIO_RATES = AudioRates(input_hz=8000, output_hz=8000)


def audio_rates_for(runner_args: RunnerArguments) -> AudioRates:
    """Telephony websockets carry 8 kHz audio; everything else uses the web rates."""
    if isinstance(runner_args, WebSocketRunnerArguments):
        return TELEPHONY_AUDIO_RATES
    return WEB_AUDIO_RATES
