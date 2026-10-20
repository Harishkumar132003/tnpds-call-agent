from pipecat.runner.types import SmallWebRTCRunnerArguments, WebSocketRunnerArguments

from voice_agent.transports import TELEPHONY_AUDIO_RATES, WEB_AUDIO_RATES, audio_rates_for


def test_websocket_runner_args_use_telephony_rates():
    args = WebSocketRunnerArguments(websocket=object(), transport_type="plivo")
    assert audio_rates_for(args) is TELEPHONY_AUDIO_RATES


def test_webrtc_runner_args_use_web_rates():
    args = SmallWebRTCRunnerArguments(webrtc_connection=object())
    assert audio_rates_for(args) is WEB_AUDIO_RATES
