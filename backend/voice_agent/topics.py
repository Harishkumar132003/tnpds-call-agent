"""Topic-aware conversation context (``LLM_TOPIC_CONTEXT=true``).

Before the agent's LLM sees a new caller turn, a small classifier call decides
whether that turn is a NEW question or a FOLLOW-UP to the current topic:

- NEW: the context sent to the LLM restarts from this caller message. Earlier
  messages stay in the call (portal, Postgres, Langfuse) but are not sent.
- FOLLOW-UP: the current context continues, however long it is.

So the LLM always sees the whole current topic and nothing else. Greetings,
thanks, "ok", "how much", "is it online" and other reactions count as
follow-ups; only a different request starts a new topic. When the classifier
fails or times out the turn is treated as a follow-up (context kept).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from loguru import logger

SARVAM_CHAT_URL = "https://api.sarvam.ai/v2/chat/completions"

CLASSIFIER_PROMPT = """\
You read a phone call to a Tamil Nadu ration card (smart card) helpline. You get the current \
topic's conversation and the caller's latest message. Decide whether the latest message starts \
a NEW topic or is a FOLLOW_UP to the current topic.
- FOLLOW_UP: it asks more about, refers to, clarifies, answers, confirms or reacts to the \
current topic. Examples: "அதுக்கு எவ்வளவு காசு?", "ஆன்லைன்ல பண்ண முடியுமா?", "எங்க போகணும்?", \
"அது கிடைச்சிருச்சுன்னா?", "சரி", "இல்ல", "நன்றி", "ஓகே", "புரியல, திரும்ப சொல்லுங்க".
- NEW: it raises a different request or subject that does not depend on the current topic. \
Example: current topic is a lost card and the caller says "என் பொண்ணு பேரை கார்டுல சேர்க்கணும்".
- Greetings, thanks, goodbyes and short reactions are FOLLOW_UP. If unsure, FOLLOW_UP.
Reply with exactly one line: the word NEW or FOLLOW_UP, a colon, then a short reason in your \
own words about this specific message (not a copy of these rules). Example: \
"FOLLOW_UP: asks the fee for the reprint just explained"."""


def _is_text(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") in ("user", "assistant")
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
    )


def caller_turn_starts(messages: list[Any]) -> list[int]:
    """Indices where a caller turn starts (consecutive caller messages = one turn: speech
    recognition often splits one utterance into pieces)."""
    starts: list[int] = []
    previous = None
    for i, m in enumerate(messages):
        if _is_text(m):
            if m["role"] == "user" and previous != "user":
                starts.append(i)
            previous = m["role"]
        elif isinstance(m, dict) and m.get("role") == "tool":
            continue  # a search result does not break the caller's turn
        else:
            previous = None
    return starts


def segment_messages(messages: list[dict], start_turn: int) -> list[dict]:
    """System messages + the opening greeting + everything from caller turn ``start_turn`` on.

    The agent's greeting (anything before the first caller turn) is always kept, so a
    restarted context still shows the model that the caller was already greeted.
    """
    head = 0
    while head < len(messages) and messages[head].get("role") == "system":
        head += 1
    starts = caller_turn_starts(messages)
    if start_turn <= 1 or not starts or start_turn > len(starts):
        return messages
    opening = [m for m in messages[head : starts[0]] if _is_text(m)]
    return messages[:head] + opening + messages[starts[start_turn - 1] :]


def _turn_text(messages: list[Any], start: int) -> str:
    pieces = []
    for m in messages[start:]:
        if _is_text(m) and m["role"] == "user":
            pieces.append(m["content"].strip())
        elif _is_text(m):
            break
    return " ".join(pieces)


class TopicTracker:
    """Per-call state: which caller turn the current topic started at."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "sarvam-105b",
        timeout: float = 2.5,
        prompt: str | None = None,
        on_call: Callable[[dict], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._timeout = timeout
        self.prompt = (prompt or "").strip() or CLASSIFIER_PROMPT
        """The classifier's system prompt (portal override, or the built-in text)."""
        self.on_call = on_call
        """Optional sink for each classifier call (request, output, usage) for Langfuse."""
        self.on_decision: Callable[[dict], None] | None = None
        """Optional sink for each decision ({turn, decision, reason, text}); the call record."""
        self.segment_start = 1
        """1-based caller turn where the current topic began."""
        self._classified = 0
        """Caller turns already classified."""
        self.decisions: list[dict] = []

    # -- deciding -------------------------------------------------------------

    async def update(self, messages: list[Any]) -> None:
        """Classify the latest caller turn once; move ``segment_start`` on a NEW topic."""
        starts = caller_turn_starts(messages)
        turns = len(starts)
        if turns == 0 or turns == self._classified:
            return
        latest = _turn_text(messages, starts[-1])
        if turns == 1:
            self._classified = turns
            self.segment_start = 1
            self._remember("NEW", "first question of the call", turns, latest)
            logger.info(f"[TOPIC] NEW question (first of the call): {latest!r}")
            return
        # The whole current topic (since the last NEW question), not a window
        current = segment_messages(messages[: starts[-1]], self.segment_start)
        lines = [
            f"{'Caller' if m['role'] == 'user' else 'Agent'}: {m['content'].strip()}"
            for m in current
            if _is_text(m)
        ]
        # Mark the turn only after a decision: Pipecat cancels this request when the caller
        # keeps talking (a split utterance), and the re-run must classify the full turn.
        decision, reason = await self._classify(lines, latest)
        self._classified = turns
        if decision == "NEW":
            self.segment_start = turns
            earlier = sum(1 for m in messages[: starts[-1]] if _is_text(m))
            logger.info(
                f"[TOPIC] >>> NEW question ({reason}) -> context restarts at caller turn "
                f"{turns}; {earlier} earlier messages no longer sent. {latest!r}"
            )
        else:
            logger.info(
                f"[TOPIC] follow-up ({reason}) -> context kept since caller turn "
                f"{self.segment_start}. {latest!r}"
            )
        self._remember(decision, reason, turns, latest)

    def _remember(self, decision: str, reason: str, turn: int, text: str) -> None:
        entry = {"turn": turn, "decision": decision, "reason": reason, "text": text}
        self.decisions.append(entry)
        if self.on_decision is not None:
            try:
                self.on_decision(entry)
            except Exception as exc:  # tracking must never break the call
                logger.warning(f"[TOPIC] could not record the decision: {exc}")

    async def _classify(self, context_lines: list[str], latest: str) -> tuple[str, str]:
        user = (
            "Current topic conversation:\n"
            + ("\n".join(context_lines) or "(none)")
            + f"\n\nCaller's latest message:\n{latest}\n\nDecision:"
        )
        body = {
            "model": self.model,
            "reasoning_effort": "none",
            "temperature": 0,
            "max_tokens": 40,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user},
            ],
        }
        started = datetime.now(UTC)
        t = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await asyncio.wait_for(
                    client.post(
                        SARVAM_CHAT_URL,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=body,
                    ),
                    timeout=self._timeout,
                )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(f"[TOPIC] classifier failed ({exc!r}); treating as follow-up")
            return "FOLLOW_UP", f"classifier failed: {type(exc).__name__}"
        text = (data["choices"][0]["message"].get("content") or "").strip()
        head, _, reason = text.partition(":")
        decision = "NEW" if head.strip().upper().startswith("NEW") else "FOLLOW_UP"
        reason = reason.strip() or text
        if self.on_call is not None:
            usage = data.get("usage") or {}
            try:
                self.on_call(
                    {
                        "label": "topic check",
                        "started_at": started.isoformat(),
                        "ended_at": datetime.now(UTC).isoformat(),
                        "model": self.model,
                        "input": body["messages"],
                        "tools": None,
                        "settings": {"model": self.model, "temperature": 0},
                        "output": text,
                        "tool_calls": [],
                        "usage": {
                            "input": int(usage.get("prompt_tokens") or 0),
                            "output": int(usage.get("completion_tokens") or 0),
                        },
                    }
                )
            except Exception as exc:  # tracing must never break the call
                logger.warning(f"[TOPIC] could not record the classifier call: {exc}")
        logger.debug(f"[TOPIC] classifier {decision} in {(time.monotonic() - t) * 1000:.0f} ms")
        return decision, reason

    # -- applying -------------------------------------------------------------

    def apply(self, messages: list[dict]) -> list[dict]:
        """The messages to send: system prompt + the current topic only."""
        segment = segment_messages(messages, self.segment_start)
        total = sum(1 for m in messages if _is_text(m))
        sent = sum(1 for m in segment if _is_text(m))
        if total > sent and segment and segment[0].get("role") == "system":
            # Earlier topics are hidden: the model must not greet again or ask who is calling
            note = (
                "\n\n[This call is already in progress: you greeted the caller at the start "
                "and earlier topics were handled. Never greet again or say your name again; "
                "reply to the caller's current message directly. If the caller refers to "
                "something from earlier that you cannot see, ask them to say it again briefly.]"
            )
            segment = [
                {**segment[0], "content": f"{segment[0].get('content') or ''}{note}"}
            ] + segment[1:]
        logger.info(
            f"[TOPIC] LLM gets {sent} of {total} messages (current topic since caller turn "
            f"{self.segment_start})" + (f" | {total - sent} earlier hidden" if total > sent else "")
        )
        return segment

    def as_json(self) -> str:
        return json.dumps(self.decisions, ensure_ascii=False)
