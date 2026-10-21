"""Send each call to Langfuse (v2) as a trace with per-turn generations.

Why not OpenTelemetry: Langfuse v2 has no OTLP endpoint, and our tracker
already knows everything worth tracing (texts, latencies, token usage,
knowledge searches). The Langfuse v2 SDK batches events on a background
thread, so nothing here waits on the network inside the audio pipeline.

Cost tracking: define model prices in Langfuse (Settings -> Models) for the
model names used here, e.g. ``sarvam-105b`` per token, ``bulbul:v3`` per
character and ``saaras:v4`` per second; Langfuse then prices every turn.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from voice_agent.config import Settings
from voice_agent.tracking.recorder import CallInfo, TurnRecord

FLUSH_TIMEOUT_SECS = 5.0


def langfuse_configured(settings: Settings) -> bool:
    """True when LANGFUSE_ENABLED is on and the host and both keys are set."""
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_host
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


class LangfuseExporter:
    """One trace per call; a generation per model call inside each turn."""

    def __init__(
        self,
        settings: Settings,
        call: CallInfo,
        call_id: uuid.UUID,
        *,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._call = call
        self._id = str(call_id)
        self._started_at = datetime.now(UTC)
        self._turns = 0
        self._transcript: list[str] = []
        self._client = client or self._make_client(settings)
        self._trace = None
        self._disabled = False

    @staticmethod
    def _make_client(settings: Settings) -> Any:
        from langfuse import Langfuse

        assert settings.langfuse_secret_key is not None
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            flush_at=25,
            flush_interval=5,
        )

    # -- sink interface (mirrors CallRecorder) --------------------------------

    def start(self) -> None:
        try:
            c = self._call
            self._trace = self._client.trace(
                id=self._id,
                name=f"{c.transport} call",
                session_id=self._id,
                timestamp=self._started_at,
                tags=[c.transport, c.language],
                metadata={
                    "transport": c.transport,
                    "direction": c.direction,
                    "from": c.from_number,
                    "to": c.to_number,
                    "language": c.language,
                    "config_id": c.config_id,
                    "plivo_call_id": c.call_id,
                },
            )
        except Exception as exc:
            self._disabled = True
            logger.warning(f"Langfuse: could not start trace: {exc}")

    def turn_completed(self, turn: TurnRecord) -> None:
        if self._disabled or self._trace is None:
            return
        try:
            self._export_turn(turn)
        except Exception as exc:
            logger.warning(f"Langfuse: turn {turn.turn_index} not exported: {exc}")

    async def close(self, end_reason: str) -> None:
        if self._disabled or self._trace is None:
            return
        try:
            self._trace.update(
                output="\n".join(self._transcript)[-4000:],
                metadata={"end_reason": end_reason, "turns": self._turns},
            )
            await asyncio.wait_for(asyncio.to_thread(self._client.flush), FLUSH_TIMEOUT_SECS)
            logger.info(f"Langfuse: trace {self._id} sent ({self._turns} turns)")
        except Exception as exc:
            logger.warning(f"Langfuse: flush failed: {exc}")

    # -- internals ------------------------------------------------------------

    def _export_turn(self, turn: TurnRecord) -> None:
        s = self._settings
        name = f"turn {turn.turn_index}"
        start = turn.user_spoke_at or turn.bot_spoke_at or turn.completed_at
        end = turn.completed_at or datetime.now(UTC)
        latency_meta = {
            "response_latency_ms": turn.response_latency_ms,
            "turn_release_ms": turn.turn_release_ms,
            "stt_ttfb_ms": turn.stt_ttfb_ms,
            "llm_ttfb_ms": turn.llm_ttfb_ms,
            "tts_ttfb_ms": turn.tts_ttfb_ms,
            "interrupted": turn.interrupted,
            "contributions": turn.contributions,
        }
        span = self._trace.span(
            name=name,
            start_time=start,
            end_time=end,
            input=turn.user_text,
            output=turn.assistant_text,
            metadata=latency_meta,
        )
        if turn.stt_audio_seconds:
            span.generation(
                name=f"{name} stt",
                model=s.sarvam_stt_model,
                start_time=start,
                end_time=end,
                output=turn.user_text,
                usage={"input": round(turn.stt_audio_seconds, 2), "unit": "SECONDS"},
            )
        for search in turn.searches:
            span.span(
                name="knowledge search",
                input=search.get("question"),
                output=[h for h in search.get("hits", []) if h.get("used")],
                metadata={
                    k: search.get(k)
                    for k in ("translated_query", "translate_ms", "embed_ms", "search_ms", "error")
                },
            )
        usage = None
        if turn.prompt_tokens is not None or turn.completion_tokens is not None:
            usage = {
                "input": turn.prompt_tokens or 0,
                "output": turn.completion_tokens or 0,
                "total": (turn.prompt_tokens or 0) + (turn.completion_tokens or 0),
                "unit": "TOKENS",
            }
        # Fixed lines (greeting, idle check, goodbye) never touch the LLM: no generation.
        llm_ran = usage is not None or bool(turn.user_text)
        if llm_ran:
            span.generation(
                name=f"{name} llm",
                model=s.llm_model,
                start_time=start,
                end_time=end,
                input=turn.user_text,
                output=turn.assistant_text,
                usage=usage,
                metadata={
                    "provider": str(s.llm_provider),
                    "reasoning_effort": s.llm_reasoning_effort,
                },
            )
        if turn.tts_characters:
            span.generation(
                name=f"{name} tts",
                model=s.sarvam_tts_model,
                start_time=start,
                end_time=end,
                output=turn.assistant_text,
                usage={"output": turn.tts_characters, "unit": "CHARACTERS"},
            )
        self._turns += 1
        if turn.user_text:
            self._transcript.append(f"User: {turn.user_text}")
        if turn.assistant_text:
            self._transcript.append(f"Assistant: {turn.assistant_text}")
