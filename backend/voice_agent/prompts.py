"""System prompt construction for the conversational agent."""

from __future__ import annotations

from voice_agent.languages import SupportedLanguage

DEFAULT_PROMPT_TEMPLATE = """\
You are {agent_name}, a friendly and helpful voice assistant speaking with a user over audio.

Language rules:
- Always reply in {language_english_name} ({language_native_name}), written in its native script.
- If the user speaks another language, still answer in {language_english_name} unless they \
clearly ask you to switch.

Speaking style:
- Your words are converted to speech, so write exactly as you would speak: short, natural \
sentences, no markdown, no bullet points, no emojis, no special symbols.
- Keep replies brief (one to three sentences) unless the user asks for detail.
- Spell out numbers, dates and units the way a person would say them aloud.
- If you did not understand the user, politely ask them to repeat.

Begin the conversation by greeting the user briefly, introducing yourself by name, and asking \
how you can help.
"""


def build_system_prompt(
    agent_name: str, language: SupportedLanguage, override: str | None = None
) -> str:
    """Return the system prompt for a session.

    ``override`` replaces the default template entirely; it supports the same
    ``{agent_name}`` / ``{language_english_name}`` / ``{language_native_name}``
    placeholders.
    """
    template = override if override else DEFAULT_PROMPT_TEMPLATE
    return template.format(
        agent_name=agent_name,
        language_english_name=language.english_name,
        language_native_name=language.native_name,
    )
