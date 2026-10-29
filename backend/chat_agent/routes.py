"""The text chat's HTTP routes, mounted by the portal under ``/api/chat``.

Separate from the portal's call and config routes: this module owns the chat
sessions and nothing else reads them. A chat lives in memory only, so it ends
when the portal restarts; what it is testing (prompts, knowledge, config
version) is stored elsewhere and read fresh for every new chat.
"""

from __future__ import annotations

import asyncpg
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from chat_agent.session import ChatSessions
from voice_agent.config import Settings
from voice_agent.knowledge.service import KnowledgeService
from voice_agent.tracking.config_store import load_active_config

MAX_MESSAGE_CHARS = 2000


class MessageIn(BaseModel):
    text: str


def create_chat_router(settings: Settings | None) -> APIRouter:
    """Routes for text chats that run a call's steps. ``settings`` None = unavailable."""
    router = APIRouter(prefix="/api/chat")
    sessions = ChatSessions()

    def agent_settings() -> Settings:
        if settings is None:
            raise HTTPException(503, "the chat needs the Sarvam key in backend/.env")
        return settings

    def pool(request: Request) -> asyncpg.Pool:
        return request.app.state.pool

    def knowledge_service(request: Request) -> KnowledgeService | None:
        """The portal's knowledge service, when searching is switched on and usable."""
        if settings is None or not settings.knowledge_enabled or settings.openai_api_key is None:
            return None
        return request.app.state.knowledge

    def session_or_404(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "chat not found or expired; start a new one")
        return session

    @router.get("/status")
    async def api_chat_status(request: Request):
        """What a new chat would run with, without starting one."""
        config = await load_active_config(pool(request))
        return {
            "available": settings is not None,
            "config_id": config.id,
            "language": (config.language or (settings.agent_language if settings else None)),
            "model": config.llm_model or (settings.llm_model if settings else None),
            "temperature": (
                config.temperature
                if config.temperature is not None
                else (settings.llm_temperature if settings else None)
            ),
            "greeting": config.greeting or (settings.agent_greeting if settings else None),
            "knowledge_on": knowledge_service(request) is not None,
            "topic_context_on": bool(settings and settings.llm_topic_context),
            "min_score": settings.knowledge_min_score if settings else None,
            "top_k": settings.knowledge_top_k if settings else None,
            "query_rewrite_on": bool(settings and settings.knowledge_query_rewrite),
        }

    @router.post("/sessions", status_code=201)
    async def api_chat_start(request: Request):
        """Start a chat on the portal's active config version."""
        config = await load_active_config(pool(request))
        session = sessions.start(
            settings=agent_settings(),
            config=config,
            knowledge=knowledge_service(request),
        )
        return session.as_dict()

    @router.get("/sessions/{session_id}")
    async def api_chat_session(session_id: str):
        return session_or_404(session_id).as_dict()

    @router.post("/sessions/{session_id}/messages")
    async def api_chat_message(session_id: str, body: MessageIn):
        """Send one message and get the agent's reply with every step it ran."""
        session = session_or_404(session_id)
        text = body.text.strip()
        if not text:
            raise HTTPException(422, "the message is empty")
        if len(text) > MAX_MESSAGE_CHARS:
            raise HTTPException(422, f"the message is longer than {MAX_MESSAGE_CHARS} characters")
        turn = await session.say(text)
        return turn.as_dict()

    @router.delete("/sessions/{session_id}")
    async def api_chat_end(session_id: str):
        return {"ended": sessions.end(session_id)}

    return router
