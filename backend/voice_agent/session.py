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
        KnowledgeTool(
            knowledge_service,
            limit=settings.knowledge_top_k,
            min_score=settings.knowledge_min_score,
            keep_best=settings.knowledge_keep_best,
            rewrite_prompt=config.rewrite_prompt,
        )
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
            exporter = LangfuseExporter(
                settings,
                info,
                call_id,
                snapshot=_debug_snapshot(settings, config, audio_rates, knowledge_tool),
            )
        tracker = ConversationTracker(
            recorder, latency_observer, extra_sinks=[exporter] if exporter else None
        )
        tracker.attach(voice_pipeline.user_aggregator, voice_pipeline.assistant_aggregator)
        if knowledge_tool is not None:
            tracker.attach_knowledge(knowledge_tool)
        voice_pipeline.worker.add_observer(tracker.usage_observer())
        if exporter is not None:  # full LLM input/output per call, for debugging in Langfuse
            from pipecat.services.llm_service import LLMService

            llm = next(
                (p for p in voice_pipeline.pipeline.processors if isinstance(p, LLMService)), None
            )
            if llm is not None and hasattr(llm, "on_request"):
                tracker.log_llm_requests()
                llm.on_request = tracker.on_llm_request
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


def _git_commit() -> str:
    """Code version for traces: APP_VERSION if set (Docker), else the git commit."""
    import os
    import subprocess

    if os.environ.get("APP_VERSION"):
        return os.environ["APP_VERSION"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if out.returncode == 0:
            return out.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return "unknown"


def _debug_snapshot(
    settings: Settings, config: AgentConfig, audio_rates: AudioRates, tool: KnowledgeTool | None
) -> dict:
    """Everything that shapes this call's behaviour, stored on its Langfuse trace."""
    import platform

    import pipecat

    return {
        "app": {
            "version": _git_commit(),
            "pipecat": getattr(pipecat, "__version__", "?"),
            "python": platform.python_version(),
        },
        "config": {
            "version": config.id,
            "language": settings.agent_language,
            "temperature": config.temperature
            if config.temperature is not None
            else settings.llm_temperature,
            "greeting": config.greeting or settings.agent_greeting,
            "custom_system_prompt": bool(config.system_prompt),
            "custom_knowledge_rules": bool(config.knowledge_prompt),
            "custom_rewrite_prompt": bool(config.rewrite_prompt),
            "idle_prompt_secs": config.idle_prompt_secs
            if config.idle_prompt_secs is not None
            else settings.idle_prompt_secs,
            "pronunciation_dictionary": config.tts_dict_id,
            "pronunciation_entries": len(config.pronunciations or []),
        },
        "models": {
            "stt": settings.sarvam_stt_model,
            "stt_mode": settings.sarvam_stt_mode,
            "llm": settings.llm_model,
            "reasoning_effort": settings.llm_reasoning_effort,
            "tts": settings.sarvam_tts_model,
            "voice": settings.sarvam_tts_speaker,
            "pace": settings.sarvam_tts_pace,
            "translate": settings.translate_model,
            "embedding": settings.embedding_model,
        },
        "knowledge": {
            "enabled": tool is not None,
            "query_rewrite": settings.knowledge_query_rewrite,
            "top_k": settings.knowledge_top_k,
            "min_score": settings.knowledge_min_score,
            "keep_best": settings.knowledge_keep_best,
            "collection": settings.qdrant_collection,
            "whole_document": settings.knowledge_whole_document,
        },
        "audio": {"input_hz": audio_rates.input_hz, "output_hz": audio_rates.output_hz},
    }


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
