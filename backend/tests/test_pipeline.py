from unittest.mock import AsyncMock, Mock

from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from voice_agent.pipeline import _register_transport_handlers, build_voice_pipeline
from voice_agent.transports import TELEPHONY_AUDIO_RATES, TRANSPORT_PARAMS
from voice_agent.transports.webrtc import webrtc_transport_params


def _offline_transport() -> SmallWebRTCTransport:
    # A connection object without an SDP offer never opens a socket.
    return SmallWebRTCTransport(SmallWebRTCConnection(), webrtc_transport_params())


def test_transport_params_are_audio_only():
    for name in ("webrtc", "plivo"):
        params = TRANSPORT_PARAMS[name]()
        assert params.audio_in_enabled and params.audio_out_enabled, name
        assert not params.video_in_enabled and not params.video_out_enabled, name


def test_pipeline_uses_web_rates_by_default(settings):
    voice_pipeline = build_voice_pipeline(_offline_transport(), settings)
    params = voice_pipeline.worker._params
    assert (params.audio_in_sample_rate, params.audio_out_sample_rate) == (16000, 24000)


def test_pipeline_uses_telephony_rates_when_asked(settings):
    voice_pipeline = build_voice_pipeline(
        _offline_transport(), settings, audio_rates=TELEPHONY_AUDIO_RATES
    )
    params = voice_pipeline.worker._params
    assert (params.audio_in_sample_rate, params.audio_out_sample_rate) == (8000, 8000)
    tts = next(
        p for p in voice_pipeline.pipeline.processors if type(p).__name__ == "SarvamTTSService"
    )
    assert tts._init_sample_rate == 8000


def test_pipeline_builds_with_system_prompt(settings):
    voice_pipeline = build_voice_pipeline(_offline_transport(), settings)
    assert voice_pipeline.worker is not None
    assert voice_pipeline.context.messages == []
    llm = next(p for p in voice_pipeline.pipeline.processors if isinstance(p, LLMService))
    assert "Hindi" in llm.settings.system_instruction
    assert "Asha" in llm.settings.system_instruction
    assert "Current date and time:" in llm.settings.system_instruction


def test_pipeline_processor_order(settings):
    voice_pipeline = build_voice_pipeline(_offline_transport(), settings)
    names = [type(p).__name__ for p in voice_pipeline.pipeline.processors]
    names = [n for n in names if n not in {"PipelineSource", "PipelineSink"}]
    assert names == [
        "SmallWebRTCInputTransport",
        "SarvamSTTService",
        "LLMUserAggregator",
        "SarvamLLMService",
        "SarvamTTSService",
        "SmallWebRTCOutputTransport",
        "LLMAssistantAggregator",
    ]


class FakeTransport:
    """Records event handlers the way BaseTransport.event_handler does."""

    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def decorator(fn):
            self.handlers[name] = fn
            return fn

        return decorator


async def test_client_connected_triggers_greeting_run():
    transport, context = FakeTransport(), LLMContext()
    worker = Mock(queue_frames=AsyncMock(), cancel=AsyncMock())
    _register_transport_handlers(transport, worker, context)

    await transport.handlers["on_client_connected"](transport, client=object())
    await transport.handlers["on_client_connected"](transport, client=object())  # duplicate event

    assert [m["role"] for m in context.messages] == ["developer"]
    assert worker.queue_frames.await_count == 1
    (frames,), _ = worker.queue_frames.await_args
    assert isinstance(frames[0], LLMRunFrame)


async def test_client_disconnected_cancels_worker():
    transport = FakeTransport()
    worker = Mock(queue_frames=AsyncMock(), cancel=AsyncMock())
    _register_transport_handlers(transport, worker, LLMContext())

    await transport.handlers["on_client_disconnected"](transport, client=object())

    worker.cancel.assert_awaited_once()


async def test_fixed_greeting_skips_the_llm():
    transport, context = FakeTransport(), LLMContext()
    worker = Mock(queue_frames=AsyncMock(), cancel=AsyncMock())
    _register_transport_handlers(transport, worker, context, greeting="வணக்கம்!")

    await transport.handlers["on_client_connected"](transport, client=object())

    assert context.messages == []  # no developer nudge, no LLM run
    (frames,), _ = worker.queue_frames.await_args
    assert isinstance(frames[0], TTSSpeakFrame) and frames[0].text == "வணக்கம்!"
