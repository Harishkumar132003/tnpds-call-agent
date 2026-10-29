"""Text chat that answers with the same steps a phone call uses.

Kept apart from :mod:`voice_agent` on purpose: nothing here runs on a call, and
the call path imports nothing from this package. See :mod:`chat_agent.session`.
"""

from chat_agent.session import ChatSession, ChatSessions, ChatTurn

__all__ = ["ChatSession", "ChatSessions", "ChatTurn"]
