"""Lifecycle of one voice session: build the pipeline, run it, clean up."""

from __future__ import annotations

import time
import uuid
from dataclasses import replace

from loguru import logger
from pipecat.observers.startup_timing_observer import (
    StartupTimingObserver,
    StartupTimingReport,
    TransportTimingReport,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.transports.base_transport import BaseTransport
from pipecat.workers.runner import WorkerRunner

from voice_agent import models as catalogue
from voice_agent.config import Settings
from voice_agent.knowledge.tool import KnowledgeTool
from voice_agent.languages import UnsupportedLanguageError, get_language
from voice_agent.pipeline import build_voice_pipeline
from voice_agent.services import build_knowledge_service
from voice_agent.tracking import get_tracking_pool
from voice_agent.tracking.config_store import AgentConfig, load_active_config
from voice_agent.tracking.conversation import ConversationTracker
from voice_agent.tracking.langfuse_export import LangfuseExporter, langfuse_configured
from voice_agent.tracking.recorder import CallInfo, CallRecorder
from voice_agent.transports import WEB_AUDIO_RATES, AudioRates


async def run_voice_session(
    transport: BaseTransport,
    settings: Settings,
    *,
    handle_sigint: bool,
    audio_rates: AudioRates = WEB_AUDIO_RATES,
    call_info: CallInfo | None = None,
) -> None:
    """Run a complete conversation on ``transport`` until the client leaves.

    ``WorkerRunner.run`` only returns once the pipeline worker has finished
    (client disconnect, idle timeout, fatal error or a signal), and the worker
    tears down every processor, including the Sarvam websockets, on the way out.

    When ``DATABASE_URL`` is set, the newest portal config (prompt, temperature)
    is applied and the conversation is recorded with per-turn latencies.
    """
    pool = await get_tracking_pool(settings.database_url)
    config = await load_active_config(pool)
    settings = apply_language_override(settings, config.language)
    settings = apply_model_overrides(settings, config)
    if call_info is not None:
        call_info = replace(call_info, language=settings.agent_language)
    latency_observer = UserBotLatencyObserver()
    startup_observer = _startup_timing_logger()
    knowledge_service = build_knowledge_service(settings)
    knowledge_tool = (
        KnowledgeTool(knowledge_service, limit=settings.knowledge_top_k)
        if knowledge_service
        else None
    )
    call_id = uuid.uuid4()  # shared by the Langfuse trace and the portal's call record
    voice_pipeline = build_voice_pipeline(
        transport,
        settings,
        audio_rates=audio_rates,
        system_prompt_override=config.system_prompt,
        temperature=config.temperature,
        greeting=config.greeting or settings.agent_greeting,
        observers=[latency_observer, startup_observer],
        knowledge=knowledge_tool,
        idle_prompt_secs=config.idle_prompt_secs,
        idle_check_line=config.idle_check_line,
        idle_goodbye_line=config.idle_goodbye_line,
        knowledge_prompt=config.knowledge_prompt,
        tts_dict_id=config.tts_dict_id,
    )

    recorder: CallRecorder | None = None
    exporter: LangfuseExporter | None = None
    if pool is not None:
        info = replace(
            call_info or CallInfo(transport="unknown", language=settings.agent_language),
            config_id=config.id,
        )
        recorder = CallRecorder(pool, info, call_id=call_id)
        if langfuse_configured(settings):
            exporter = LangfuseExporter(settings, info, call_id)
        tracker = ConversationTracker(
            recorder, latency_observer, extra_sinks=[exporter] if exporter else None
        )
        tracker.attach(voice_pipeline.user_aggregator, voice_pipeline.assistant_aggregator)
        if knowledge_tool is not None:
            tracker.attach_knowledge(knowledge_tool)
        voice_pipeline.worker.add_observer(tracker.usage_observer())
        recorder.start()
        if exporter is not None:
            exporter.start()

    runner = WorkerRunner(handle_sigint=handle_sigint)
    started = time.monotonic()
    logger.info(
        f"Session starting (language={settings.agent_language}, "
        f"stt={settings.sarvam_stt_model}, tts={settings.sarvam_tts_model}, "
        f"llm={settings.llm_provider}/{settings.llm_model}, "
        f"audio={audio_rates.input_hz}/{audio_rates.output_hz} Hz, "
        f"config={'db#' + str(config.id) if config.id else 'env'}, "
        f"tracking={'on' if recorder else 'off'}, "
        f"knowledge={'on' if knowledge_tool else 'off'}, "
        f"tracing={'langfuse' if langfuse_configured(settings) else 'off'}, call={call_id})"
    )
    end_reason = "completed"
    try:
        await runner.add_workers(voice_pipeline.worker)
        await runner.run()
    except BaseException as exc:
        end_reason = f"error: {type(exc).__name__}"
        raise
    finally:
        end_reason = voice_pipeline.state.get("end_reason", end_reason)
        duration = time.monotonic() - started
        turns = _count_user_turns(voice_pipeline.context)
        logger.info(f"Session ended after {duration:.1f}s with {turns} user turn(s)")
        if recorder is not None:
            await recorder.close(end_reason)
        if exporter is not None:
            await exporter.close(end_reason)


def _startup_timing_logger() -> StartupTimingObserver:
    """Log how long each processor took to start, so slow first turns can be explained."""
    observer = StartupTimingObserver()

    @observer.event_handler("on_startup_timing_report")
    async def on_startup(_: object, report: StartupTimingReport) -> None:
        slow = sorted(report.processor_timings, key=lambda t: -t.duration_secs)[:4]
        parts = ", ".join(f"{t.processor_name} {t.duration_secs * 1000:.0f} ms" for t in slow)
        logger.info(
            f"Pipeline started in {report.total_duration_secs * 1000:.0f} ms; slowest: {parts}"
        )

    @observer.event_handler("on_transport_timing_report")
    async def on_transport(_: object, report: TransportTimingReport) -> None:
        if report.client_connected_secs is not None:
            connected_ms = report.client_connected_secs * 1000
            logger.info(f"Client connected {connected_ms:.0f} ms after pipeline start")

    return observer


def apply_model_overrides(settings: Settings, config: AgentConfig) -> Settings:
    """Return settings with the portal's per-step model choices applied.

    Choices that no longer fit the catalogue or the language are ignored with a
    warning, so a stale config version cannot stop calls from being answered.
    """
    chosen = {
        "sarvam_stt_model": config.stt_model,
        "llm_model": config.llm_model,
        "sarvam_tts_model": config.tts_model,
        "sarvam_tts_speaker": config.tts_speaker,
        "translate_model": config.translate_model,
        "embedding_model": config.embedding_model,
    }
    update = {k: v for k, v in chosen.items() if v}
    if config.embedding_model:
        update["embedding_dimensions"] = None  # resolved from the model's known size
    if not update:
        return settings
    try:
        catalogue.validate_selection(
            settings.agent_language,
            stt_model=config.stt_model,
            llm_model=config.llm_model,
            tts_model=config.tts_model or settings.sarvam_tts_model,
            tts_speaker=config.tts_speaker,
            translate_model=config.translate_model,
            embedding_model=config.embedding_model,
        )
    except ValueError as exc:
        logger.warning(f"Ignoring portal model choices: {exc}")
        return settings
    return settings.model_copy(update=update)


def apply_language_override(settings: Settings, language: str | None) -> Settings:
    """Return settings with the portal's language applied, if it is a supported one."""
    if not language or language == settings.agent_language:
        return settings
    try:
        get_language(language)
    except UnsupportedLanguageError as exc:
        logger.warning(f"Ignoring portal language: {exc}")
        return settings
    return settings.model_copy(update={"agent_language": language})


def _count_user_turns(context: LLMContext) -> int:
    return sum(1 for m in context.messages if isinstance(m, dict) and m.get("role") == "user")
