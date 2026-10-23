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
import contextlib
import uuid
from datetime import UTC, datetime, timedelta
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


def _parse_time(value: object):
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


class LangfuseExporter:
    """One trace per call; a generation per model call inside each turn."""

    def __init__(
        self,
        settings: Settings,
        call: CallInfo,
        call_id: uuid.UUID,
        *,
        client: Any | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        self._snapshot = snapshot or {}
        self._log_sink: int | None = None
        self._log_counts = {"WARNING": 0, "ERROR": 0, "CRITICAL": 0}
        self._log_repeats: dict[tuple[str, str], int] = {}
        """(level, message) -> times seen; repeats are counted, not stored again."""
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
                input={"call": c.__dict__, **self._snapshot},
                metadata={
                    "transport": c.transport,
                    "direction": c.direction,
                    "from": c.from_number,
                    "to": c.to_number,
                    "language": c.language,
                    "config_id": c.config_id,
                    "plivo_call_id": c.call_id,
                    **self._snapshot,
                },
            )
        except Exception as exc:
            self._disabled = True
            logger.warning(f"Langfuse: could not start trace: {exc}")

    # -- the call's own log lines, as trace events -----------------------------

    def attach_logs(self) -> None:
        """Send this call's log lines to the trace as events.

        Lines are matched by ``call_id`` in loguru's context (the session runs inside
        ``logger.contextualize(call_id=...)``), so concurrent calls never mix. Our own
        code is kept from INFO up; libraries (Pipecat, Sarvam, httpx) from WARNING up.
        """
        if self._disabled or self._trace is None or self._log_sink is not None:
            return

        def sink(message: Any) -> None:
            record = message.record
            if record["extra"].get("call_id") != self._id:
                return
            name = record["name"] or ""
            if name.startswith("voice_agent.tracking.langfuse_export"):
                return  # never trace our own tracing
            level = record["level"].name
            ours = name.startswith(("voice_agent", "__main__", "bot"))
            if record["level"].no < (20 if ours else 30):
                return
            if level in self._log_counts:
                self._log_counts[level] += 1
            text = record["message"]
            key = (level, text)
            self._log_repeats[key] = self._log_repeats.get(key, 0) + 1
            if self._log_repeats[key] > 1:
                return  # stored once; the count goes on the trace at the end
            with contextlib.suppress(Exception):  # tracing must never break logging
                self._trace.event(
                    name=f"log {level.lower()}: {text[:80]}",
                    start_time=record["time"],
                    level="ERROR"
                    if level in ("ERROR", "CRITICAL")
                    else "WARNING"
                    if level == "WARNING"
                    else "DEFAULT",
                    status_message=text if level != "INFO" else None,
                    input=text,
                    metadata={
                        "logger": name,
                        "function": record["function"],
                        "line": record["line"],
                        "exception": str(record["exception"]) if record["exception"] else None,
                    },
                )

        self._log_sink = logger.add(sink, level="DEBUG", enqueue=False, catch=True)

    def detach_logs(self) -> None:
        if self._log_sink is not None:
            with contextlib.suppress(ValueError):
                logger.remove(self._log_sink)
            self._log_sink = None

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
            self.detach_logs()
            ended = datetime.now(UTC)
            repeated = {
                f"{level}: {text}": count
                for (level, text), count in self._log_repeats.items()
                if count > 1
            }
            if repeated:
                self._trace.event(
                    name=f"log lines repeated ({len(repeated)} distinct)",
                    start_time=ended,
                    level="WARNING",
                    output=dict(sorted(repeated.items(), key=lambda kv: -kv[1])),
                )
            self._trace.update(
                output="\n".join(self._transcript),
                metadata={
                    "end_reason": end_reason,
                    "turns": self._turns,
                    "duration_secs": round((ended - self._started_at).total_seconds(), 1),
                    "warnings": self._log_counts["WARNING"],
                    "distinct_warnings": sum(
                        1 for level, _ in self._log_repeats if level == "WARNING"
                    ),
                    "errors": self._log_counts["ERROR"] + self._log_counts["CRITICAL"],
                },
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
        heard = [_parse_time(m.get("at")) for m in turn.user_messages]
        for m in turn.user_messages:  # each separate utterance, before they were merged
            span.event(name="caller said", start_time=_parse_time(m.get("at")), input=m.get("text"))
        if turn.interrupted:
            span.event(
                name="agent interrupted by caller",
                start_time=end,
                level="WARNING",
                output=turn.assistant_text,
                metadata={"spoken_before_interruption": turn.assistant_text},
            )
        if turn.stt_audio_seconds:
            span.generation(
                name=f"{name} stt",
                model=s.sarvam_stt_model,
                start_time=start,
                end_time=(heard[-1] if heard and heard[-1] else end),
                output=turn.user_text,
                usage={"input": round(turn.stt_audio_seconds, 2), "unit": "SECONDS"},
                metadata={
                    "utterances": turn.user_messages,
                    "mode": s.sarvam_stt_mode,
                    "language": s.agent_language,
                },
            )
        for search in turn.searches:
            s_start = _parse_time(search.get("started_at")) or start
            s_end = _parse_time(search.get("at")) or end
            hits = search.get("hits", [])
            searched_for = (
                search.get("rewritten_query")
                or search.get("translated_query")
                or search.get("question")
            )
            span.span(
                name="knowledge search",
                start_time=s_start,
                end_time=s_end,
                level="WARNING" if search.get("error") else "DEFAULT",
                status_message=search.get("error"),
                input={
                    "caller_text": search.get("question"),
                    "searched_for": searched_for,
                    "how": "rewrite"
                    if search.get("rewritten_query")
                    else "translation"
                    if search.get("translated_query")
                    else "as is",
                    "settings": search.get("settings"),
                },
                output={
                    "sent": [h for h in hits if h.get("used")],
                    "not_sent": [
                        {k: h.get(k) for k in ("source", "chunk_index", "score")}
                        for h in hits
                        if not h.get("used")
                    ],
                },
                metadata={
                    k: search.get(k)
                    for k in (
                        "rewrite_ms",
                        "translate_ms",
                        "embed_ms",
                        "search_ms",
                        "translated_query",
                        "rewritten_query",
                        "error",
                    )
                },
            )
            # Sarvam bills translation per input character; costed via the model price
            question = search.get("question") or ""
            if search.get("translated_query") and question:
                t_start = _parse_time(search.get("started_at")) or start or end
                span.generation(
                    name=f"{name} translate",
                    model=s.translate_model,
                    start_time=t_start,
                    end_time=t_start + timedelta(milliseconds=search.get("translate_ms") or 0),
                    input=question,
                    output=search.get("translated_query"),
                    usage={"input": len(question), "unit": "CHARACTERS"},
                    metadata={"translate_ms": search.get("translate_ms")},
                )
            # Query rewrite: an LLM call, costed per token
            prompt_tokens, completion_tokens = (search.get("rewrite_tokens") or [0, 0])[:2]
            if search.get("rewritten_query") and prompt_tokens + completion_tokens:
                r_start = _parse_time(search.get("started_at")) or start or end
                span.generation(
                    name=f"{name} rewrite",
                    model=s.llm_model,
                    start_time=r_start,
                    end_time=r_start + timedelta(milliseconds=search.get("rewrite_ms") or 0),
                    input=search.get("rewrite_input") or question,
                    output=search.get("rewritten_query"),
                    usage={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                        "unit": "TOKENS",
                    },
                    metadata={"rewrite_ms": search.get("rewrite_ms")},
                )
        usage = None
        if turn.prompt_tokens is not None or turn.completion_tokens is not None:
            usage = {
                "input": turn.prompt_tokens or 0,
                "output": turn.completion_tokens or 0,
                "total": (turn.prompt_tokens or 0) + (turn.completion_tokens or 0),
                "unit": "TOKENS",
            }
        # Full request/response of every LLM call (system prompt, all messages, tools)
        for n, call in enumerate(turn.llm_calls, start=1):
            calls_usage = call.get("usage") or {}
            output = call.get("output") or None
            if call.get("tool_calls"):
                output = {"text": output, "tool_calls": call["tool_calls"]}
            label = (
                " (tool call)"
                if call.get("tool_calls")
                else " (interrupted, no reply)"
                if not call.get("output") and not calls_usage
                else ""
            )
            span.generation(
                name=f"{name} llm {n}{label}",
                model=call.get("model") or s.llm_model,
                start_time=_parse_time(call.get("started_at")) or start,
                end_time=_parse_time(call.get("ended_at")) or end,
                input=call.get("input"),
                output=output,
                usage={
                    "input": calls_usage.get("input", 0),
                    "output": calls_usage.get("output", 0),
                    "total": calls_usage.get("input", 0) + calls_usage.get("output", 0),
                    "unit": "TOKENS",
                }
                if calls_usage
                else None,
                model_parameters=call.get("settings") or None,
                metadata={"tools": call.get("tools")},
            )
        # Fixed lines (greeting, idle check, goodbye) never touch the LLM: no generation.
        llm_ran = (usage is not None or bool(turn.user_text)) and not turn.llm_calls
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
                start_time=turn.bot_spoke_at or start,
                end_time=end,
                input=turn.assistant_text,
                output=turn.assistant_text,
                usage={"output": turn.tts_characters, "unit": "CHARACTERS"},
                metadata={
                    "voice": s.sarvam_tts_speaker,
                    "pace": s.sarvam_tts_pace,
                    "language": s.agent_language,
                    "first_audio_ms": turn.tts_ttfb_ms,
                    "interrupted": turn.interrupted,
                },
            )
        self._turns += 1
        if turn.user_text:
            self._transcript.append(f"User: {turn.user_text}")
        if turn.assistant_text:
            self._transcript.append(f"Assistant: {turn.assistant_text}")
