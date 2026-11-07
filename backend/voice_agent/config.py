"""Application configuration loaded from environment variables / ``.env``."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_agent.languages import get_language

SarvamSTTMode = Literal["transcribe", "translate", "verbatim", "translit", "codemix"]


class LLMProvider(StrEnum):
    """Which chat-completion backend generates the agent's replies."""

    SARVAM = "sarvam"
    OPENAI = "openai"


class ConfigurationError(RuntimeError):
    """Raised when the environment is missing or has invalid settings."""


class Settings(BaseSettings):
    """All runtime settings. Field names map to upper-cased environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Sarvam
    sarvam_api_key: SecretStr
    agent_language: str = "hi-IN"
    stt_auto_detect_language: bool = False
    sarvam_stt_model: str = "saaras:v4"
    sarvam_stt_mode: SarvamSTTMode = "transcribe"
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str | None = None
    sarvam_tts_pace: float = Field(default=1.0, gt=0)  # model range checked in services
    translate_model: str = "mayura:v1"
    """Sarvam translation model for knowledge-base queries
    (``mayura:v1`` or ``sarvam-translate:v1``)."""

    # LLM
    llm_provider: LLMProvider = LLMProvider.SARVAM
    llm_model: str = "sarvam-105b"
    llm_temperature: float | None = Field(default=None, ge=0, le=2)
    llm_topic_context: bool = False
    """Before each reply, a small LLM check decides if the caller's turn is a NEW question
    (context restarts there) or a FOLLOW-UP (context continues). Off = whole conversation."""
    topic_check_timeout_secs: float = Field(default=2.5, gt=0)
    llm_reasoning_effort: Literal["none", "low", "medium", "high"] | None = "none"
    llm_stall_retry_secs: float = Field(default=5.0, ge=0)
    """On a call, re-issue the answer request once when nothing has arrived after this many
    seconds (the second attempt is not cut off). 0 turns it off. Rate limits (429), 5xx and
    connection errors are retried twice by the HTTP client regardless."""
    """Sarvam's reasoning models think before answering, which adds 1-2 s to every reply.
    "none" turns that off (measured: first word in ~0.3 s instead of 1-2.4 s)."""
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None

    # Deployment: how the bot is reached from outside (used when the CLI flags are absent)
    public_host: str | None = None
    """Public hostname Plivo connects to (wss://<host>/ws); becomes the runner's --proxy."""
    bot_transport: str | None = None
    """Restrict the runner to one transport (``plivo``, ``webrtc``). Unset = all at once."""
    bot_public_url: str = "http://localhost:7860"
    """Where a browser reaches the bot for test calls (the portal's Test call page)."""
    """Runner transport to start with, e.g. "plivo"; becomes the runner's -t flag."""

    # Plivo telephony (optional; also read directly by the Pipecat runner)
    plivo_auth_id: str | None = None
    plivo_auth_token: SecretStr | None = None
    plivo_from_number: str | None = None
    mobile_number: str | None = None

    # Agent behaviour
    agent_name: str = "Asha"
    agent_system_prompt: str | None = None
    agent_greeting: str | None = None
    """Fixed opening line (in the agent's language) spoken instantly; empty = LLM greets."""

    # Tracking / portal (optional). Example: postgresql://user:pass@localhost:5432/voice_agent
    database_url: str | None = None
    portal_host: str = "127.0.0.1"
    """Bind address for the portal; 0.0.0.0 inside containers / behind a proxy."""
    portal_port: int = 8090

    # Langfuse (optional): one trace per call with per-turn usage, for cost and evaluation.
    langfuse_enabled: bool = False
    """Master switch: LANGFUSE_ENABLED=false stops sending traces without removing the keys."""
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None

    # Knowledge base (portal uploads). Embeddings use OPENAI_API_KEY.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "voice_agent_knowledge_v2"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None
    embedding_base_url: str | None = None
    knowledge_enabled: bool = True
    """Let the agent search uploaded documents (needs OPENAI_API_KEY and Qdrant)."""
    knowledge_whole_document: bool = True
    """Index each uploaded document as one chunk (one FAQ sheet = one topic)."""
    knowledge_top_k: int = Field(default=5, ge=1, le=10)
    """Most sheets one question's search hands to the LLM."""
    knowledge_min_score: float = Field(default=0.40, ge=0, le=1)
    """Send every sheet whose similarity reaches this score; below it nothing is sent."""
    knowledge_query_rewrite: bool = False
    """Rewrite each knowledge question + recent conversation into an English query first."""
    knowledge_search_type: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    """How calls, test calls and the chat search the sheets: semantic (embeddings only),
    keyword (BM25 only) or hybrid (both, merged by reciprocal rank fusion). Keyword and hybrid
    send every retrieved sheet up to KNOWLEDGE_TOP_K; KNOWLEDGE_MIN_SCORE applies to semantic."""
    knowledge_rewrite_timeout_secs: float = Field(default=3.0, gt=0)
    knowledge_chunk_chars: int = Field(default=1200, gt=100)
    knowledge_chunk_overlap: int = Field(default=200, ge=0)

    # Idle handling: after the agent finishes speaking and the caller stays silent for
    # IDLE_PROMPT_SECS, ask once whether they are still there; if silence continues for
    # the same period, say goodbye and end the call. 0 disables.
    idle_prompt_secs: float = Field(default=10.0, ge=0)
    idle_check_line: str | None = None
    """Override for the "are you still there?" line (default is per language)."""
    idle_goodbye_line: str | None = None
    """Override for the closing line spoken before hanging up (default is per language)."""

    # Runtime
    log_level: str = "INFO"
    session_idle_timeout_secs: float = Field(default=300.0, gt=0)

    @field_validator("sarvam_tts_speaker", "openai_base_url", "agent_system_prompt", mode="before")
    @classmethod
    def _blank_string_is_none(cls, value: object) -> object:
        """Treat ``KEY=`` lines in ``.env`` as unset rather than as empty strings."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openai_api_key", "plivo_auth_token", mode="before")
    @classmethod
    def _blank_secret_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("agent_language")
    @classmethod
    def _language_must_be_supported(cls, value: str) -> str:
        get_language(value)  # raises UnsupportedLanguageError with the supported list
        return value

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid LOG_LEVEL {value!r}")
        return level

    @model_validator(mode="after")
    def _openai_provider_needs_key(self) -> Settings:
        if self.llm_provider is LLMProvider.OPENAI and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return self


def load_settings() -> Settings:
    """Load and validate settings, converting pydantic errors into a readable message."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'settings'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(f"Invalid configuration: {problems}") from exc
