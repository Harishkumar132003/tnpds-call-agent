"""One text chat, run through the same four steps as a phone call.

This module is deliberately separate from the call path: it has no Pipecat
pipeline, no speech recognition and no text to speech, and nothing in
``voice_agent`` imports it. What it shares with a call is only data and library
pieces, so the two cannot drift apart:

* the portal's active config version (persona, per-step prompts, temperature,
  language, greeting, model choices),
* the same :class:`~voice_agent.topics.TopicTracker`,
* the same :class:`~voice_agent.knowledge.tool.KnowledgeTool`, so retrieval,
  the top-k limit and the score threshold are identical,
* the same built-in prompt text (``ROUTER_PROMPT`` / ``ANSWER_PROMPT``).

One caller message runs the steps a call runs, in the same order:

1. **topic check** - NEW or FOLLOW_UP, which decides the messages the LLM sees.
2. **LLM 1, decide** - persona + router prompt, with the search tool offered.
   Any text it writes alongside a tool call is dropped, exactly as on a call.
3. **search** - the knowledge tool: query rewrite, embedding, Qdrant, then the
   same selection rules (top-k, minimum score).
4. **LLM 2, answer** - persona + answer prompt, grounded in the returned sheets.

The reply, every step and every search come back on :class:`ChatTurn`, so the
portal can show what the agent did rather than just what it said.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
from loguru import logger

from voice_agent import models as catalogue
from voice_agent.config import Settings
from voice_agent.knowledge.service import KnowledgeService
from voice_agent.knowledge.tool import (
    ANSWER_PROMPT,
    ROUTER_PROMPT,
    KnowledgeTool,
    SearchRecord,
    searching_line,
)
from voice_agent.languages import UnsupportedLanguageError, get_language
from voice_agent.prompts import build_system_prompt
from voice_agent.topics import TopicTracker
from voice_agent.tracking.config_store import AgentConfig

SARVAM_CHAT_URL = "https://api.sarvam.ai/v2/chat/completions"
MAX_REPLY_TOKENS = 800
REQUEST_TIMEOUT_SECS = 90.0
MAX_SEARCH_ROUNDS = 2
"""Searches allowed in one chat turn, as on a call: the first decide step and one follow-up."""


def _time_note() -> str:
    """The same note the call's prompt carries, so the model never guesses the date."""
    now = datetime.now().astimezone()
    return f"\n\nCurrent date and time: {now.strftime('%A, %d %B %Y, %H:%M %Z')}.\n"


def _is_text(message: dict) -> bool:
    return bool(message.get("content")) and message.get("role") in ("user", "assistant")


class _ToolCall:
    """Stand-in for Pipecat's ``FunctionCallParams``.

    The knowledge tool only reads ``arguments``, ``context.messages`` and calls
    ``result_callback``, so the chat can run the call's own ``handle()`` without
    a pipeline behind it.
    """

    def __init__(self, name: str, arguments: dict, messages: list[dict]) -> None:
        self.function_name = name
        self.arguments = arguments
        self.tool_call_id = "chat"
        self.llm = None
        self.context = SimpleNamespace(messages=messages)
        self.result: Any = None

    async def result_callback(self, result: Any, **_: Any) -> None:
        self.result = result


@dataclass
class _Reply:
    """One chat-completion response."""

    text: str
    tool_calls: list[dict]
    prompt_tokens: int
    completion_tokens: int
    ms: int
    finish: str | None


@dataclass
class ChatTurn:
    """Everything one caller message produced."""

    index: int
    user_text: str
    reply: str
    searched: bool
    filler: str | None
    topic: dict | None
    steps: list[dict]
    searches: list[dict]
    dropped_text: str | None
    prompt_tokens: int
    completion_tokens: int
    ms: int
    error: str | None = None
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "user_text": self.user_text,
            "reply": self.reply,
            "searched": self.searched,
            "filler": self.filler,
            "topic": self.topic,
            "steps": self.steps,
            "searches": self.searches,
            "dropped_text": self.dropped_text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ms": self.ms,
            "error": self.error,
            "at": self.at,
        }


class ChatSession:
    """A chat with its own conversation, topic state and prompts."""

    def __init__(
        self,
        *,
        settings: Settings,
        config: AgentConfig,
        knowledge: KnowledgeService | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now(UTC)
        self.used_at = self.created_at
        self.config_id = config.id
        self.language = _language_of(settings, config)
        self.model = config.llm_model or settings.llm_model
        self.temperature = (
            config.temperature if config.temperature is not None else settings.llm_temperature
        )
        self.reasoning_effort = catalogue.reasoning_effort_for(
            self.model, settings.llm_reasoning_effort
        )
        self._api_key = settings.sarvam_api_key.get_secret_value()
        self._lock = asyncio.Lock()
        self.messages: list[dict] = []
        """The whole conversation, in chat-completion shape (no system message)."""
        self.turns: list[ChatTurn] = []

        persona = (
            build_system_prompt(
                settings.agent_name,
                self.language,
                override=config.system_prompt or settings.agent_system_prompt,
            )
            + _time_note()
        )
        # Each step gets rules for its own job, composed exactly as the call does.
        self.tool: KnowledgeTool | None = (
            KnowledgeTool(
                knowledge,
                limit=settings.knowledge_top_k,
                min_score=settings.knowledge_min_score,
                rewrite_prompt=config.rewrite_prompt,
            )
            if knowledge is not None
            else None
        )
        self.router_prompt = persona
        self.answer_prompt = persona
        self.tools: list[dict] | None = None
        if self.tool is not None:
            router = (config.router_prompt or ROUTER_PROMPT).strip()
            answer = (config.answer_prompt or ANSWER_PROMPT).strip()
            self.router_prompt = f"{persona}\n\n{router}\n"
            self.answer_prompt = f"{persona}\n\n{answer}\n"
            self.tools = [
                {"type": "function", "function": schema.to_default_dict()}
                for schema in self.tool.schema().standard_tools
            ]
            self.tool.on_search(self._remember_search)

        self.topics: TopicTracker | None = None
        if settings.llm_topic_context:
            self.topics = TopicTracker(
                self._api_key,
                model=self.model,
                timeout=settings.topic_check_timeout_secs,
                prompt=config.topic_prompt,
            )
            if self.tool is not None:
                self.tool.topics = self.topics  # the query rewrite sees the current topic only

        self.greeting = (config.greeting or settings.agent_greeting or "").strip() or None
        if self.greeting:
            # A call speaks the fixed greeting before the LLM runs; the chat shows it.
            self.messages.append({"role": "assistant", "content": self.greeting})
        self._searches: list[SearchRecord] = []

    # -- info -----------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language.code,
            "language_name": self.language.english_name,
            "greeting": self.greeting,
            "config_id": self.config_id,
            "model": self.model,
            "temperature": self.temperature,
            "knowledge_on": self.tool is not None,
            "topic_context_on": self.topics is not None,
            "min_score": self.tool.min_score if self.tool else None,
            "top_k": self.tool.limit if self.tool else None,
            "started_at": self.created_at.isoformat(),
            "turns": [t.as_dict() for t in self.turns],
        }

    # -- one turn -------------------------------------------------------------

    async def say(self, text: str) -> ChatTurn:
        """Run the caller's message through every step and return the agent's reply."""
        async with self._lock:
            return await self._say(text)

    async def _say(self, text: str) -> ChatTurn:
        started = time.monotonic()
        self.used_at = datetime.now(UTC)
        self._searches = []
        self.messages.append({"role": "user", "content": text})
        steps: list[dict] = []
        prompt_tokens = completion_tokens = 0
        dropped: list[str] = []
        filler: str | None = None
        error: str | None = None

        topic = await self._topic_check(steps)

        reply = ""
        rounds = 0
        while True:
            after_tool = bool(self.messages) and self.messages[-1].get("role") == "tool"
            system = self.answer_prompt if after_tool else self.router_prompt
            label = "llm answer" if after_tool else "llm decide"
            messages = self._context(system)
            try:
                answer = await self._complete(messages, tools=self.tools)
            except Exception as exc:
                logger.warning(f"[CHAT] {label} failed: {exc!r}")
                error = f"{label} failed: {exc}"
                steps.append({"step": label, "ms": 0, "error": str(exc)[:300]})
                break
            prompt_tokens += answer.prompt_tokens
            completion_tokens += answer.completion_tokens
            step = {
                "step": label,
                "ms": answer.ms,
                "messages_sent": sum(1 for m in messages if _is_text(m)),
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "searched": bool(answer.tool_calls),
                "finish": answer.finish,
            }

            if answer.tool_calls and rounds >= MAX_SEARCH_ROUNDS:
                logger.info("[CHAT] ignoring a further search in this turn; answering instead")
                step["note"] = "search limit reached, answered from what was already found"
                steps.append(step)
                final = await self._complete(self._context(system), tools=None)
                prompt_tokens += final.prompt_tokens
                completion_tokens += final.completion_tokens
                reply = final.text
                steps.append({"step": "llm answer", "ms": final.ms, "forced_answer": True})
                break

            if answer.tool_calls:
                if answer.text.strip():
                    # The call holds this text back and drops it: only the reply
                    # written after the search results may reach the caller.
                    dropped.append(answer.text.strip())
                    logger.info(f"[CHAT] LLM 1 text dropped (tool call follows): {answer.text!r}")
                step["questions"] = [
                    call.get("arguments") for call in answer.tool_calls if call.get("arguments")
                ]
                steps.append(step)
                if filler is None:
                    filler = searching_line(self.language.code)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call["raw"] for call in answer.tool_calls],
                    }
                )
                for call in answer.tool_calls:
                    steps.append(await self._search(call))
                rounds += 1
                continue

            reply = answer.text
            steps.append(step)
            break

        if reply.strip():
            self.messages.append({"role": "assistant", "content": reply})
        elif error is None:
            error = "the model returned no text"
            logger.warning("[CHAT] the model returned an empty reply")

        turn = ChatTurn(
            index=len(self.turns) + 1,
            user_text=text,
            reply=reply,
            searched=bool(self._searches),
            filler=filler if self._searches else None,
            topic=topic,
            steps=steps,
            searches=[record.as_dict() for record in self._searches],
            dropped_text="\n".join(dropped) or None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ms=round((time.monotonic() - started) * 1000),
            error=error,
        )
        self.turns.append(turn)
        logger.info(
            f"[CHAT] turn {turn.index} in {turn.ms} ms: "
            f"{'searched' if turn.searched else 'no search'}, "
            f"{len(turn.searches)} search(es), reply {len(turn.reply)} chars"
        )
        return turn

    # -- steps ----------------------------------------------------------------

    async def _topic_check(self, steps: list[dict]) -> dict | None:
        """Step 1: NEW or FOLLOW_UP for the caller's latest message."""
        if self.topics is None:
            return None
        before = self.topics.segment_start
        started = time.monotonic()
        try:
            await self.topics.update(self._all_messages())
        except Exception as exc:  # a failed check keeps the context as it is
            logger.warning(f"[CHAT] topic check failed, context kept: {exc!r}")
            steps.append({"step": "topic check", "ms": 0, "error": str(exc)[:300]})
            return None
        decision = self.topics.decisions[-1] if self.topics.decisions else None
        steps.append(
            {
                "step": "topic check",
                "ms": round((time.monotonic() - started) * 1000),
                "decision": (decision or {}).get("decision"),
                "reason": (decision or {}).get("reason"),
                "restarted": self.topics.segment_start != before,
                "segment_start": self.topics.segment_start,
            }
        )
        return decision

    async def _search(self, call: dict) -> dict:
        """Step 3: the call's own knowledge tool, then the result as a tool message."""
        assert self.tool is not None
        started = time.monotonic()
        before = len(self._searches)
        context = self._context(self.router_prompt)
        params = _ToolCall(call["name"], call.get("arguments") or {}, context)
        try:
            await self.tool.handle(params)
            result = params.result
        except Exception as exc:
            logger.warning(f"[CHAT] search failed: {exc!r}")
            result = {"found": False, "reason": f"search failed: {exc}"}
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
        found = bool(isinstance(result, dict) and result.get("found"))
        passages = (result or {}).get("passages") or [] if isinstance(result, dict) else []
        return {
            "step": "search",
            "ms": round((time.monotonic() - started) * 1000),
            "found": found,
            "reason": (result or {}).get("reason") if isinstance(result, dict) else None,
            "questions": [q.get("question") for q in (result or {}).get("questions") or []]
            if isinstance(result, dict)
            else [],
            "sheets_sent": len(passages),
            "sheets_retrieved": sum(len(r.hits) for r in self._searches[before:]),
            "sources": [p.get("source") for p in passages],
        }

    def _remember_search(self, record: SearchRecord) -> None:
        self._searches.append(record)

    # -- plumbing -------------------------------------------------------------

    def _all_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.router_prompt}, *self.messages]

    def _context(self, system: str) -> list[dict]:
        """The messages to send: system prompt + the current topic only (as on a call)."""
        messages = [{"role": "system", "content": system}, *self.messages]
        if self.topics is None:
            return messages
        return self.topics.apply(messages)

    async def _complete(self, messages: list[dict], *, tools: list[dict] | None) -> _Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_REPLY_TOKENS,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECS) as client:
            response = await client.post(
                SARVAM_CHAT_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
            )
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return _Reply(
            text=(message.get("content") or "").strip(),
            tool_calls=_tool_calls(message),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            ms=round((time.monotonic() - started) * 1000),
            finish=choice.get("finish_reason"),
        )


def _tool_calls(message: dict) -> list[dict]:
    """The model's tool calls, with their arguments parsed."""
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            raw_arguments = function.get("arguments")
            logger.warning(f"[CHAT] could not read the tool arguments: {raw_arguments!r}")
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {"questions": arguments}
        calls.append(
            {
                "id": raw.get("id") or "chat",
                "name": function.get("name") or "",
                "arguments": arguments,
                "raw": raw,
            }
        )
    return calls


def _language_of(settings: Settings, config: AgentConfig):
    """The chat's language: the portal's choice when it is a supported one."""
    code = config.language or settings.agent_language
    try:
        return get_language(code)
    except UnsupportedLanguageError as exc:
        logger.warning(f"[CHAT] ignoring the portal language: {exc}")
        return get_language(settings.agent_language)


class ChatSessions:
    """The chats held in memory, newest kept, idle ones dropped."""

    def __init__(self, *, limit: int = 20, idle_secs: float = 7200) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._limit = limit
        self._idle_secs = idle_secs

    def start(
        self, *, settings: Settings, config: AgentConfig, knowledge: KnowledgeService | None
    ) -> ChatSession:
        self._sweep()
        session = ChatSession(settings=settings, config=config, knowledge=knowledge)
        self._sessions[session.id] = session
        logger.info(
            f"[CHAT] new chat {session.id[:8]} on config #{session.config_id}, "
            f"{session.language.code}, knowledge {'on' if session.tool else 'off'}, "
            f"topic context {'on' if session.topics else 'off'}"
        )
        return session

    def get(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def end(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _sweep(self) -> None:
        now = datetime.now(UTC)
        for key, session in list(self._sessions.items()):
            if (now - session.used_at).total_seconds() > self._idle_secs:
                del self._sessions[key]
        while len(self._sessions) >= self._limit:
            oldest = min(self._sessions.values(), key=lambda s: s.used_at)
            del self._sessions[oldest.id]
