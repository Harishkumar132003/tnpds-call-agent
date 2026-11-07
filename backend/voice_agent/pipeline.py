"""Construction of the Pipecat voice pipeline.

Pipeline (audio flows top to bottom, context flows back up through the
assistant aggregator)::

    transport.input()      microphone audio from the client
      -> Sarvam STT        streaming transcription (websocket)
      -> user aggregator   VAD + turn detection, appends user turn to context
      -> LLM               generates the reply text (streamed)
      -> Sarvam TTS        streaming speech synthesis (websocket)
      -> transport.output() audio back to the client
      -> assistant aggregator  appends the spoken reply to context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMRunFrame, TTSSpeakFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.transports.base_transport import BaseTransport

from voice_agent.config import Settings
from voice_agent.fillers import FillerClips, Voice, load_filler_clips
from voice_agent.knowledge.tool import (
    ANSWER_PROMPT,
    ROUTER_PROMPT,
    TOOL_NAME,
    KnowledgeTool,
    searching_line,
)
from voice_agent.languages import get_language
from voice_agent.phrases import idle_check_line as default_idle_check_line
from voice_agent.phrases import idle_goodbye_line as default_idle_goodbye_line
from voice_agent.prompts import build_system_prompt
from voice_agent.services import build_llm, build_stt, build_tts
from voice_agent.topics import TopicTracker
from voice_agent.transports import WEB_AUDIO_RATES, AudioRates


@dataclass(frozen=True)
class VoicePipeline:
    """The pipeline worker plus handles useful for tests and diagnostics."""

    worker: PipelineWorker
    pipeline: Pipeline
    context: LLMContext
    user_aggregator: LLMUserAggregator
    assistant_aggregator: LLMAssistantAggregator
    state: dict[str, str] = field(default_factory=dict)
    """Mutable session facts set by handlers, e.g. ``end_reason``."""


def build_voice_pipeline(
    transport: BaseTransport,
    settings: Settings,
    *,
    audio_rates: AudioRates = WEB_AUDIO_RATES,
    system_prompt_override: str | None = None,
    temperature: float | None = None,
    greeting: str | None = None,
    observers: list[BaseObserver] | None = None,
    knowledge: KnowledgeTool | None = None,
    idle_prompt_secs: float | None = None,
    idle_check_line: str | None = None,
    idle_goodbye_line: str | None = None,
    router_prompt: str | None = None,
    answer_prompt: str | None = None,
    topic_prompt: str | None = None,
    tts_dict_id: str | None = None,
) -> VoicePipeline:
    """Wire the Sarvam STT/TTS services, the LLM and the transport into a worker.

    ``system_prompt_override`` / ``temperature`` (from the portal's stored
    config) take precedence over the environment settings when given. A
    ``greeting`` is spoken straight through TTS when the client connects,
    skipping the LLM for the opening line; without one the LLM greets.
    """
    language = get_language(settings.agent_language)
    idle_secs = settings.idle_prompt_secs if idle_prompt_secs is None else idle_prompt_secs
    system_prompt = (
        build_system_prompt(
            settings.agent_name,
            language,
            override=system_prompt_override or settings.agent_system_prompt,
        )
        + _current_time_note()
    )
    base_prompt = system_prompt
    if knowledge is not None:
        # Each LLM step gets rules for its own job: deciding (and searching) vs
        # writing the grounded answer from the results. Same persona, same model.
        system_prompt = base_prompt + "\n\n" + (router_prompt or ROUTER_PROMPT).strip() + "\n"
    stt = build_stt(settings)
    tts = build_tts(settings, sample_rate=audio_rates.output_hz, dict_id=tts_dict_id)
    llm = build_llm(settings, system_prompt, temperature=temperature)
    if knowledge is not None and hasattr(llm, "answer_instruction"):
        llm.answer_instruction = (
            base_prompt + "\n\n" + (answer_prompt or ANSWER_PROMPT).strip() + "\n"
        )
    if settings.llm_topic_context and hasattr(llm, "topics"):
        topics = TopicTracker(
            settings.sarvam_api_key.get_secret_value(),
            model=settings.llm_model,
            timeout=settings.topic_check_timeout_secs,
            prompt=topic_prompt,
        )
        llm.topics = topics  # type: ignore[attr-defined]
        if knowledge is not None:
            knowledge.topics = topics  # the query rewrite sees the current topic only

    context = LLMContext(tools=knowledge.schema()) if knowledge is not None else LLMContext()
    if knowledge is not None:
        clips = None
        if settings.filler_clips:
            voice = Voice(
                model=settings.sarvam_tts_model,
                speaker=settings.sarvam_tts_speaker,
                pace=settings.sarvam_tts_pace,
                dict_id=tts_dict_id,
            )
            clips = load_filler_clips(language.code, voice)
        _register_knowledge_tool(
            llm, tts, knowledge, language.code, clips=clips, sample_rate=audio_rates.output_hz
        )
    aggregators = LLMContextAggregatorPair(
        context,
        # Local Silero VAD drives turn start (and barge-in interruptions); the
        # default stop strategy (bundled Smart Turn v3) decides end of turn.
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # Fires on_user_turn_idle when the caller stays silent after the agent
            # finishes speaking (see _register_idle_handling).
            user_idle_timeout=idle_secs,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=audio_rates.input_hz,
            audio_out_sample_rate=audio_rates.output_hz,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=settings.session_idle_timeout_secs,
        observers=observers,
        # If a service can no longer work (e.g. Sarvam rejects the API key and
        # the websocket cannot be re-established) end the session instead of
        # leaving the user talking to a silent bot.
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    # SarvamSTTService registers these events but never emits them in 1.12.0
    # (only SarvamRealtimeSTTService does), so only the TTS service is wired up.
    _log_service_connection_events(tts, "Sarvam TTS")
    _log_conversation_turns(aggregators.user(), aggregators.assistant())
    _register_transport_handlers(transport, worker, context, greeting=greeting)
    state: dict[str, str] = {}
    if idle_secs > 0:
        _register_idle_handling(
            aggregators.user(),
            worker,
            state,
            check_line=idle_check_line
            or settings.idle_check_line
            or default_idle_check_line(language.code),
            goodbye_line=idle_goodbye_line
            or settings.idle_goodbye_line
            or default_idle_goodbye_line(language.code),
        )
    return VoicePipeline(
        worker=worker,
        pipeline=pipeline,
        context=context,
        user_aggregator=aggregators.user(),
        assistant_aggregator=aggregators.assistant(),
        state=state,
    )


def _register_knowledge_tool(
    llm: object,
    tts: object,
    tool: KnowledgeTool,
    language: str,
    *,
    clips: FillerClips | None = None,
    sample_rate: int = WEB_AUDIO_RATES.output_hz,
) -> None:
    """Expose the search function to the LLM and say a filler line while it runs.

    The line is played from ``clips`` when it was recorded with the session's
    voice, and synthesised otherwise. Either way it is queued through the TTS
    service so it keeps its place in front of the answer's audio.
    """
    llm.register_function(TOOL_NAME, tool.handle)  # type: ignore[attr-defined]
    if hasattr(llm, "hold_text_before_tools"):
        llm.hold_text_before_tools = True  # only the reply after the search answers

    @llm.event_handler("on_function_calls_started")  # type: ignore[attr-defined]
    async def on_function_calls_started(_service: object, function_calls: list) -> None:
        if not any(getattr(call, "function_name", "") == TOOL_NAME for call in function_calls):
            return
        filler = searching_line(language)  # a different line each search
        frames = await clips.frames(filler, sample_rate) if clips is not None else None
        if frames:
            logger.info(f"Searching knowledge base; playing the recorded line: {filler}")
            for frame in frames:
                await tts.queue_frame(frame)  # type: ignore[attr-defined]
        else:
            logger.info(f"Searching knowledge base; telling the caller: {filler}")
            await tts.queue_frame(TTSSpeakFrame(filler, append_to_context=False))  # type: ignore[attr-defined]


def _current_time_note() -> str:
    """Appended to the prompt so the model never guesses today's date."""
    now = datetime.now().astimezone()
    return f"\n\nCurrent date and time: {now.strftime('%A, %d %B %Y, %H:%M %Z')}.\n"


def _register_idle_handling(
    user: LLMUserAggregator,
    worker: PipelineWorker,
    state: dict[str, str],
    *,
    check_line: str,
    goodbye_line: str,
) -> None:
    """Silence after the agent speaks: ask once, then say goodbye and end the call.

    The aggregator's idle timer starts each time the agent stops speaking and is
    cancelled when the caller speaks, so the check line itself restarts the
    countdown; a second timeout means the caller never replied.
    """
    idle_strikes = 0

    @user.event_handler("on_user_turn_started")
    async def on_user_turn_started(_aggregator: object, *_: object) -> None:
        nonlocal idle_strikes
        idle_strikes = 0

    @user.event_handler("on_user_turn_idle")
    async def on_user_turn_idle(_aggregator: object) -> None:
        nonlocal idle_strikes
        idle_strikes += 1
        if idle_strikes == 1:
            logger.info("Caller idle; asking once whether they are still there")
            await worker.queue_frames([TTSSpeakFrame(check_line)])
            return
        logger.info("Caller still idle after the check; saying goodbye and ending the call")
        state["end_reason"] = "idle_timeout"
        # EndFrame is a control frame: the goodbye finishes playing before the
        # pipeline shuts down (and, on Plivo, hangs up the call).
        await worker.queue_frames([TTSSpeakFrame(goodbye_line), EndFrame()])


def _register_transport_handlers(
    transport: BaseTransport,
    worker: PipelineWorker,
    context: LLMContext,
    *,
    greeting: str | None = None,
) -> None:
    greeted = False

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport: BaseTransport, client: object) -> None:
        nonlocal greeted
        if greeted:  # SmallWebRTC can report "connected" more than once per peer
            return
        greeted = True
        if greeting:
            # Spoken immediately; TTSSpeakFrame appends it to the context as the
            # assistant's first turn so the LLM knows what was said.
            logger.info("Client connected; speaking fixed greeting")
            await worker.queue_frames([TTSSpeakFrame(greeting)])
            return
        logger.info("Client connected; starting conversation")
        # Same pattern as Pipecat's own Sarvam example: nudge the LLM to open
        # the conversation, then run it once.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport: BaseTransport, client: object) -> None:
        logger.info("Client disconnected; ending session")
        await worker.cancel()


def _log_service_connection_events(service: object, label: str) -> None:
    """Surface websocket lifecycle of a Sarvam service in the logs."""

    @service.event_handler("on_connected")  # type: ignore[attr-defined]
    async def on_connected(_service: object) -> None:
        logger.info(f"{label}: connected")

    @service.event_handler("on_disconnected")  # type: ignore[attr-defined]
    async def on_disconnected(_service: object) -> None:
        logger.info(f"{label}: disconnected")

    @service.event_handler("on_connection_error")  # type: ignore[attr-defined]
    async def on_connection_error(_service: object, error: object) -> None:
        logger.error(f"{label}: connection error: {error}")


def _log_conversation_turns(user: LLMUserAggregator, assistant: LLMAssistantAggregator) -> None:
    """Log each completed user and assistant turn at INFO level."""

    @user.event_handler("on_user_turn_message_added")
    async def on_user_turn(_aggregator: object, message: UserTurnMessageAddedMessage) -> None:
        logger.info(f"User: {message.content}")

    @assistant.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn(_aggregator: object, message: AssistantTurnStoppedMessage) -> None:
        if not message.content.strip():
            return  # function-call round; the spoken answer is logged when it arrives
        suffix = " (interrupted)" if message.interrupted else ""
        logger.info(f"Assistant{suffix}: {message.content}")
