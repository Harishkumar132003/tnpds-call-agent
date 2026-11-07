from unittest.mock import AsyncMock, Mock

from pipecat.frames.frames import EndFrame, LLMRunFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from voice_agent.pipeline import (
    _register_idle_handling,
    _register_transport_handlers,
    build_voice_pipeline,
)
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
        p
        for p in voice_pipeline.pipeline.processors
        if type(p).__name__ == "SarvamTTSWithDictionary"
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
        "SarvamCatalogueLLMService",
        "SarvamTTSWithDictionary",
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


async def test_idle_asks_once_then_ends_the_call():
    user = FakeTransport()  # same event_handler shape as the aggregator
    worker = Mock(queue_frames=AsyncMock())
    state: dict[str, str] = {}
    _register_idle_handling(user, worker, state, check_line="still there?", goodbye_line="bye")  # type: ignore[arg-type]

    await user.handlers["on_user_turn_idle"](user)
    (frames,), _ = worker.queue_frames.await_args
    assert [type(f) for f in frames] == [TTSSpeakFrame] and frames[0].text == "still there?"
    assert state == {}

    await user.handlers["on_user_turn_idle"](user)
    (frames,), _ = worker.queue_frames.await_args
    assert [type(f) for f in frames] == [TTSSpeakFrame, EndFrame] and frames[0].text == "bye"
    assert state["end_reason"] == "idle_timeout"


async def test_idle_counter_resets_when_the_caller_speaks():
    user = FakeTransport()
    worker = Mock(queue_frames=AsyncMock())
    state: dict[str, str] = {}
    _register_idle_handling(user, worker, state, check_line="still there?", goodbye_line="bye")  # type: ignore[arg-type]
    await user.handlers["on_user_turn_idle"](user)
    await user.handlers["on_user_turn_started"](user, None)  # caller replied
    await user.handlers["on_user_turn_idle"](user)  # a later silence: asks again, does not hang up
    (frames,), _ = worker.queue_frames.await_args
    assert [type(f) for f in frames] == [TTSSpeakFrame] and state == {}


def test_idle_disabled_when_zero(make_settings):
    voice_pipeline = build_voice_pipeline(_offline_transport(), make_settings(idle_prompt_secs=0))
    assert voice_pipeline.state == {}


def test_idle_overrides_from_portal_config(settings):
    voice_pipeline = build_voice_pipeline(
        _offline_transport(), settings, idle_prompt_secs=25, idle_check_line="custom check"
    )
    params = voice_pipeline.user_aggregator._params
    assert params.user_idle_timeout == 25
    off = build_voice_pipeline(_offline_transport(), settings, idle_prompt_secs=0)
    assert off.user_aggregator._params.user_idle_timeout == 0


def test_each_llm_step_gets_its_own_prompt(make_settings):
    from unittest.mock import Mock

    from pipecat.services.llm_service import LLMService

    from voice_agent.knowledge.tool import ANSWER_PROMPT, ROUTER_PROMPT, KnowledgeTool

    tool = KnowledgeTool(service=Mock())
    default = build_voice_pipeline(_offline_transport(), make_settings(), knowledge=tool)
    llm = next(p for p in default.pipeline.processors if isinstance(p, LLMService))
    # LLM 1 (deciding/search) rules in the service prompt, LLM 2 (answering) rules on the side
    assert ROUTER_PROMPT.strip() in llm.settings.system_instruction
    assert ANSWER_PROMPT.strip() not in llm.settings.system_instruction
    assert ANSWER_PROMPT.strip() in llm.answer_instruction
    assert ROUTER_PROMPT.strip() not in llm.answer_instruction
    # both share the persona text
    persona = llm.settings.system_instruction.split(ROUTER_PROMPT.strip())[0]
    assert llm.answer_instruction.startswith(persona.rstrip("\n").rstrip())

    overridden = build_voice_pipeline(
        _offline_transport(),
        make_settings(),
        knowledge=tool,
        router_prompt="Always search first.",
        answer_prompt="Answer only from sheets.",
    )
    llm = next(p for p in overridden.pipeline.processors if isinstance(p, LLMService))
    assert "Always search first." in llm.settings.system_instruction
    assert ROUTER_PROMPT.strip() not in llm.settings.system_instruction
    assert "Answer only from sheets." in llm.answer_instruction

    without = build_voice_pipeline(_offline_transport(), make_settings())
    llm = next(p for p in without.pipeline.processors if isinstance(p, LLMService))
    assert ROUTER_PROMPT.strip() not in llm.settings.system_instruction
    assert llm.answer_instruction is None


class _FakeLLM:
    def __init__(self):
        self.handlers = {}

    def register_function(self, name, handler):
        pass

    def event_handler(self, name):
        def register(fn):
            self.handlers[name] = fn
            return fn

        return register


async def test_search_filler_plays_the_recorded_clip_or_falls_back_to_tts(tmp_path):
    import json
    import wave
    from types import SimpleNamespace

    from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

    from voice_agent.fillers import FillerClips
    from voice_agent.knowledge.tool import SEARCHING_LINES, TOOL_NAME, KnowledgeTool
    from voice_agent.pipeline import _register_knowledge_tool

    clips = []
    for n, text in enumerate(SEARCHING_LINES["ta-IN"]):
        with wave.open(str(tmp_path / f"{n}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x01\x00" * 24000)
        clips.append({"file": f"{n}.wav", "text": text})
    manifest = {"language": "ta-IN", "model": "m", "speaker": "s", "pace": 1.0, "clips": clips}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    recorded = FillerClips.load("ta-IN", tmp_path)
    search = [SimpleNamespace(function_name=TOOL_NAME)]

    llm, tts = _FakeLLM(), Mock(queue_frame=AsyncMock())
    _register_knowledge_tool(
        llm, tts, KnowledgeTool(service=Mock()), "ta-IN", clips=recorded, sample_rate=8000
    )
    await llm.handlers["on_function_calls_started"](llm, search)
    frames = [call.args[0] for call in tts.queue_frame.await_args_list]
    assert isinstance(frames[0], TTSStartedFrame) and isinstance(frames[-1], TTSStoppedFrame)
    assert all(isinstance(f, TTSAudioRawFrame) and f.sample_rate == 8000 for f in frames[1:-1])
    assert not any(isinstance(f, TTSSpeakFrame) for f in frames)

    tts.queue_frame.reset_mock()
    await llm.handlers["on_function_calls_started"](llm, [SimpleNamespace(function_name="other")])
    tts.queue_frame.assert_not_awaited()  # only the search gets a filler

    llm, tts = _FakeLLM(), Mock(queue_frame=AsyncMock())
    _register_knowledge_tool(llm, tts, KnowledgeTool(service=Mock()), "ta-IN", clips=None)
    await llm.handlers["on_function_calls_started"](llm, search)
    (frame,), _ = tts.queue_frame.await_args
    assert isinstance(frame, TTSSpeakFrame) and frame.append_to_context is False
    assert frame.text in SEARCHING_LINES["ta-IN"]
