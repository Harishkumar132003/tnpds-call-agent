"""Assembles per-turn transcripts and latencies from pipeline events.

Sources (all event callbacks, no frame inspection of our own):

- ``LLMUserAggregator.on_user_turn_message_added``: the user's text.
- ``UserBotLatencyObserver``: user-silence to bot-speech latency, with
  per-service TTFB breakdown, and the greeting's connect-to-speech latency.
- ``LLMAssistantAggregator.on_assistant_turn_stopped``: the assistant's text,
  which closes the turn.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData, STTUsageMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown, UserBotLatencyObserver
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMUserAggregator,
    UserTurnMessageAddedMessage,
)

from voice_agent.knowledge.tool import KnowledgeTool, SearchRecord
from voice_agent.tracking.recorder import CallRecorder, TurnRecord


def _ms(seconds: float | None) -> int | None:
    return None if seconds is None else round(seconds * 1000)


class TurnSink(Protocol):
    """Anything that wants finished turns: the Postgres recorder, Langfuse, ..."""

    def turn_completed(self, turn: TurnRecord) -> None: ...


class ConversationTracker:
    """Collects one :class:`TurnRecord` per assistant turn and hands it to every sink."""

    def __init__(
        self,
        recorder: CallRecorder,
        latency_observer: UserBotLatencyObserver,
        *,
        extra_sinks: list[TurnSink] | None = None,
    ) -> None:
        self._sinks: list[TurnSink] = [recorder, *(extra_sinks or [])]
        self._turn = TurnRecord(turn_index=0)
        self._register_latency_events(latency_observer)

    def usage_observer(self) -> BaseObserver:
        """Pipeline observer that attributes token/character/second usage to the current turn."""
        tracker = self

        class _UsageObserver(BaseObserver):
            """Observers see a frame at every processor-to-processor hop; count it once."""

            def __init__(self) -> None:
                super().__init__()
                self._seen: dict[int, None] = {}

            async def on_push_frame(self, data: FramePushed) -> None:
                if not isinstance(data.frame, MetricsFrame):
                    return
                if data.frame.id in self._seen:
                    return
                self._seen[data.frame.id] = None
                if len(self._seen) > 2000:  # bounded memory for long calls
                    for key in list(self._seen)[:1000]:
                        del self._seen[key]
                for item in data.frame.data:
                    tracker._add_usage(item)

        return _UsageObserver()

    def _add_usage(self, item: object) -> None:
        turn = self._turn
        if isinstance(item, LLMUsageMetricsData):
            turn.prompt_tokens = (turn.prompt_tokens or 0) + (item.value.prompt_tokens or 0)
            turn.completion_tokens = (turn.completion_tokens or 0) + (
                item.value.completion_tokens or 0
            )
        elif isinstance(item, TTSUsageMetricsData):
            turn.tts_characters = (turn.tts_characters or 0) + int(item.value)
        elif isinstance(item, STTUsageMetricsData):
            turn.stt_audio_seconds = (turn.stt_audio_seconds or 0.0) + float(
                item.value.audio_seconds
            )

    def attach_knowledge(self, tool: KnowledgeTool) -> None:
        """Keep every search (and its chunks) with the turn that triggered it."""

        def on_search(record: SearchRecord) -> None:
            self._turn.searches.append(record.as_dict())

        tool.on_search(on_search)

    def attach(self, user: LLMUserAggregator, assistant: LLMAssistantAggregator) -> None:
        @user.event_handler("on_user_turn_message_added")
        async def on_user_turn(_: object, message: UserTurnMessageAddedMessage) -> None:
            # A caller may be split into several user messages before the agent
            # answers (end-of-turn fired mid-thought); keep all of them.
            text = message.content.strip()
            if self._turn.user_text:
                self._turn.user_text = f"{self._turn.user_text} {text}"
            else:
                self._turn.user_text = text
                self._turn.user_spoke_at = datetime.now(UTC)

        @assistant.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn(_: object, message: AssistantTurnStoppedMessage) -> None:
            if not message.content.strip() and not message.interrupted:
                return  # a function-call round with no spoken text; the answer follows
            self._turn.assistant_text = message.content
            self._turn.interrupted = message.interrupted
            self._turn.completed_at = datetime.now(UTC)
            for sink in self._sinks:
                sink.turn_completed(self._turn)
            self._turn = TurnRecord(turn_index=self._turn.turn_index + 1)

    def _register_latency_events(self, observer: UserBotLatencyObserver) -> None:
        @observer.event_handler("on_first_bot_speech_latency")
        async def on_first_speech(_: object, latency: float) -> None:
            self._turn.response_latency_ms = _ms(latency)
            self._turn.bot_spoke_at = datetime.now(UTC)

        @observer.event_handler("on_latency_measured")
        async def on_latency(_: object, latency: float) -> None:
            self._turn.response_latency_ms = _ms(latency)
            self._turn.bot_spoke_at = datetime.now(UTC)

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(_: object, breakdown: LatencyBreakdown) -> None:
            self._turn.turn_release_ms = _ms(breakdown.user_turn_secs)
            self._turn.contributions = [
                {
                    "key": c.key,
                    "label": c.label,
                    "owner": c.owner,
                    "ms": _ms(c.duration_secs),
                }
                for c in breakdown.contributions
            ]
            for ttfb in breakdown.ttfb:
                name = ttfb.processor.upper()
                if "STT" in name:
                    self._turn.stt_ttfb_ms = _ms(ttfb.duration_secs)
                elif "LLM" in name:
                    self._turn.llm_ttfb_ms = _ms(ttfb.duration_secs)
                elif "TTS" in name:
                    self._turn.tts_ttfb_ms = _ms(ttfb.duration_secs)
