"""Factories for the Sarvam speech services and the LLM used by the pipeline.

Each factory only *constructs* a Pipecat service; no network connection is made
until the pipeline starts, which keeps these functions cheap, unit-testable and
usable as a start-up preflight check.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import TTS_MODEL_CONFIGS, SarvamTTSService

from voice_agent import models as catalogue
from voice_agent.config import ConfigurationError, LLMProvider, Settings
from voice_agent.languages import get_language
from voice_agent.prompts import build_system_prompt

if TYPE_CHECKING:
    from voice_agent.knowledge.service import KnowledgeService


def build_stt(settings: Settings) -> SarvamSTTService:
    """Create the streaming Sarvam speech-to-text service."""
    language = (
        None
        if settings.stt_auto_detect_language
        else get_language(settings.agent_language).pipecat_language
    )
    return SarvamSTTService(
        api_key=settings.sarvam_api_key.get_secret_value(),
        mode=settings.sarvam_stt_mode,
        settings=SarvamSTTService.Settings(
            model=settings.sarvam_stt_model,
            language=language,
        ),
    )


def _register_v4_flash() -> None:
    """Add bulbul:v4-flash to Pipecat's TTS model table (Pipecat 1.12 stops at v3).

    Same request shape as v3 (checked over the websocket on 2026-10-22, with
    a pronunciation dictionary); only the voice ids differ.
    """
    if "bulbul:v4-flash" in TTS_MODEL_CONFIGS:
        return
    v3 = TTS_MODEL_CONFIGS["bulbul:v3"]
    TTS_MODEL_CONFIGS["bulbul:v4-flash"] = dataclasses.replace(
        v3,
        default_speaker="shubh_enhi_ads",
        speakers=tuple(catalogue.speakers_for("bulbul:v4-flash")),
    )


_register_v4_flash()


class SarvamTTSWithDictionary(SarvamTTSService):
    """SarvamTTSService plus a pronunciation dictionary (``dict_id`` in the websocket config).

    Pipecat 1.12 builds the config message without ``dict_id``; this keeps its
    payload and adds the one field. Revisit when upstream exposes it.
    """

    def __init__(self, *, dict_id: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dict_id = dict_id

    def _config_data(self) -> dict:
        data = {
            "target_language_code": self._settings.language,
            "speaker": self._settings.voice,
            "speech_sample_rate": self._speech_sample_rate,
            "enable_preprocessing": self._settings.enable_preprocessing,
            "min_buffer_size": self._settings.min_buffer_size,
            "max_chunk_length": self._settings.max_chunk_length,
            "output_audio_codec": self._output_audio_codec,
            "output_audio_bitrate": self._output_audio_bitrate,
            "pace": self._settings.pace,
            "model": self._settings.model,
        }
        for key in ("pitch", "loudness", "temperature"):
            value = getattr(self._settings, key)
            if value is not None:
                data[key] = value
        if self.dict_id:
            data["dict_id"] = self.dict_id
        return data

    async def _send_config(self) -> None:
        if not self._websocket:
            raise Exception("WebSocket not connected")
        import json

        config = self._config_data()
        await self._websocket.send(json.dumps({"type": "config", "data": config}))


def build_tts(
    settings: Settings, sample_rate: int | None = None, dict_id: str | None = None
) -> SarvamTTSService:
    """Create the streaming Sarvam text-to-speech service.

    ``sample_rate`` lets telephony sessions ask Sarvam for 8 kHz directly
    instead of synthesising at 24 kHz and downsampling. ``dict_id`` applies a
    Sarvam pronunciation dictionary (see :mod:`voice_agent.pronunciation`).
    """
    language = get_language(settings.agent_language)
    _validate_tts_options(settings)
    tts_settings = SarvamTTSService.Settings(
        model=settings.sarvam_tts_model,
        language=language.pipecat_language,
        pace=settings.sarvam_tts_pace,
    )
    if settings.sarvam_tts_speaker:
        tts_settings.voice = settings.sarvam_tts_speaker
    return SarvamTTSWithDictionary(
        api_key=settings.sarvam_api_key.get_secret_value(),
        sample_rate=sample_rate,
        settings=tts_settings,
        dict_id=dict_id,
    )


def _validate_tts_options(settings: Settings) -> None:
    """Reject speaker/pace values Pipecat would otherwise silently ignore or clamp."""
    model = settings.sarvam_tts_model
    config = TTS_MODEL_CONFIGS.get(model)
    if config is None:
        allowed = ", ".join(sorted(TTS_MODEL_CONFIGS))
        raise ConfigurationError(
            f"SARVAM_TTS_MODEL={model!r} is not supported. Choose one of: {allowed}"
        )
    low, high = config.pace_range
    if not low <= settings.sarvam_tts_pace <= high:
        raise ConfigurationError(
            f"SARVAM_TTS_PACE={settings.sarvam_tts_pace} is outside {model}'s range {low}-{high}"
        )
    speaker = settings.sarvam_tts_speaker
    if speaker and speaker not in catalogue.speakers_for(model):
        raise ConfigurationError(
            f"SARVAM_TTS_SPEAKER={speaker!r} is not available for {model}. "
            f"Choose one of: {', '.join(catalogue.speakers_for(model))}"
        )


class OpenAICompatibleLLMService(OpenAILLMService):
    """``OpenAILLMService`` for third-party endpoints that reject the ``developer`` role.

    Pipecat's adapter rewrites ``developer`` messages to ``user`` when this is
    False, which is what its own OpenRouter/Ollama services do.
    """

    supports_developer_role = False


class SarvamCatalogueLLMService(SarvamLLMService):
    """SarvamLLMService that accepts the models Sarvam's API serves today.

    Pipecat 1.12 ships a fixed list (glm5.2, deepseekv4-flash, ...) that lags
    the API (glm5.3, deepseekv4.1-flash, ...). The catalogue in
    :mod:`voice_agent.models` is the source of truth; every model there accepts
    ``reasoning_effort`` (checked 2026-10-21), so all are treated as reasoning
    models and get the setting from :func:`voice_agent.models.reasoning_effort_for`.
    """

    _SUPPORTED_MODELS = frozenset(m.id for m in catalogue.LLM_MODELS)
    _REASONING_MODELS = frozenset(m.id for m in catalogue.LLM_MODELS)

    on_request = None
    """Optional ``callable(params)`` given each exact chat request (Langfuse LLM logging)."""
    topics = None
    """Optional :class:`voice_agent.topics.TopicTracker` (LLM_TOPIC_CONTEXT)."""
    answer_instruction = None
    """System prompt for the answering step. A request whose last message is a tool result
    is writing the caller's answer from search results (LLM 2): it gets this prompt instead
    of the deciding/search prompt the service was built with (LLM 1)."""
    hold_text_before_tools = False
    """True when tools are registered: a reply to the caller (LLM 1) only decides whether to
    call a tool. Its text is held until the reply ends and dropped when it calls a tool, so
    the answer always comes from the reply after the tool result (LLM 2)."""
    _held: list[str] | None = None

    async def _process_context(self, context):
        messages = context.messages
        last = messages[-1] if messages else None
        after_tool = isinstance(last, dict) and last.get("role") == "tool"
        self._held = [] if self.hold_text_before_tools and not after_tool else None
        try:
            await super()._process_context(context)
        finally:
            held, self._held = self._held, None
        if held:  # no tool call: the reply is the answer, speak it now
            await super()._push_llm_text("".join(held))

    async def _push_llm_text(self, text: str):
        if self._held is not None:
            self._held.append(text)
            return
        await super()._push_llm_text(text)

    async def run_function_calls(self, function_calls):
        if function_calls and self._held is not None:
            if "".join(self._held).strip():
                logger.info(f"LLM 1 text dropped (tool call follows): {''.join(self._held)!r}")
            self._held = None
        await super().run_function_calls(function_calls)

    async def get_chat_completions(self, context):
        if self.topics is not None:
            try:
                await self.topics.update(list(context.messages))
            except Exception as exc:  # a failed check keeps the context as it is
                logger.warning(f"[TOPIC] check failed, context kept: {exc}")
        return await super().get_chat_completions(context)

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        messages = params.get("messages")
        if (
            self.answer_instruction
            and isinstance(messages, list)
            and messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "tool"
            and messages[0].get("role") == "system"
        ):
            params["messages"] = [{**messages[0], "content": self.answer_instruction}] + messages[
                1:
            ]
        if self.topics is not None and isinstance(params.get("messages"), list):
            params["messages"] = self.topics.apply(params["messages"])
        if self.on_request is not None:
            try:
                self.on_request(params)
            except Exception as exc:  # logging must never break a reply
                logger.warning(f"LLM request logging failed: {exc}")
        return params


def build_llm(
    settings: Settings, system_instruction: str, temperature: float | None = None
) -> LLMService:
    """Create the chat LLM service for the configured provider.

    The system prompt is passed as the service's ``system_instruction`` rather
    than as a message in the context, as recommended by Pipecat 1.x.
    ``temperature`` ``None`` keeps the provider default.
    """
    if temperature is None:
        temperature = settings.llm_temperature
    stall = {
        "retry_on_timeout": settings.llm_stall_retry_secs > 0,
        "retry_timeout_secs": settings.llm_stall_retry_secs or None,
    }
    if settings.llm_provider is LLMProvider.SARVAM:
        return SarvamCatalogueLLMService(
            api_key=settings.sarvam_api_key.get_secret_value(),
            **stall,
            settings=SarvamLLMService.Settings(
                model=settings.llm_model,
                system_instruction=system_instruction,
                temperature=temperature,
                reasoning_effort=catalogue.reasoning_effort_for(
                    settings.llm_model, settings.llm_reasoning_effort
                ),
            ),
        )
    assert settings.openai_api_key is not None  # enforced by Settings
    service_class = (
        OpenAILLMService if settings.openai_base_url is None else OpenAICompatibleLLMService
    )
    return service_class(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        **stall,
        settings=OpenAILLMService.Settings(
            model=settings.llm_model,
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )


def validate_service_configuration(settings: Settings) -> None:
    """Build every service once, offline, so bad models/speakers/prompts fail at startup.

    Raises:
        ConfigurationError: with a message naming the offending setting.
    """
    language = get_language(settings.agent_language)
    try:
        prompt = build_system_prompt(
            settings.agent_name, language, override=settings.agent_system_prompt
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ConfigurationError(
            f"AGENT_SYSTEM_PROMPT has an invalid placeholder: {exc}. "
            "Only {agent_name}, {language_english_name} and {language_native_name} are allowed."
        ) from exc
    try:
        build_stt(settings)
        build_tts(settings)
        build_llm(settings, prompt)
    except ValueError as exc:  # pipecat rejects unknown models with ValueError
        raise ConfigurationError(str(exc)) from exc


def build_query_rewriter(settings: Settings):
    """The context-aware query rewriter when KNOWLEDGE_QUERY_REWRITE is on, else None."""
    if not settings.knowledge_query_rewrite:
        return None
    from voice_agent.knowledge.rewrite import QueryRewriter

    return QueryRewriter(
        settings.sarvam_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout=settings.knowledge_rewrite_timeout_secs,
    )


def build_knowledge_service(
    settings: Settings, *, search_type: str | None = None
) -> KnowledgeService | None:
    """Retrieval for the agent; ``None`` when disabled or the embedding key is missing.
    ``search_type`` overrides KNOWLEDGE_SEARCH_TYPE (evaluation runs pick their own)."""
    search_type = search_type or settings.knowledge_search_type
    if not settings.knowledge_enabled or settings.openai_api_key is None:
        return None
    from voice_agent.knowledge.embeddings import OpenAIEmbedder
    from voice_agent.knowledge.service import KnowledgeService
    from voice_agent.knowledge.store import KnowledgeStore
    from voice_agent.knowledge.translate import SarvamTranslator

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        base_url=settings.embedding_base_url,
    )
    store = KnowledgeStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        collection=settings.qdrant_collection,
        vector_size=embedder.dimensions,
    )
    return KnowledgeService(
        pool=None,
        store=store,
        embedder=embedder,
        rewriter=build_query_rewriter(settings),
        translator=SarvamTranslator(
            settings.sarvam_api_key.get_secret_value(),
            source_language=settings.agent_language,
            model=settings.translate_model,
        ),
        chunk_chars=settings.knowledge_chunk_chars,
        chunk_overlap=settings.knowledge_chunk_overlap,
        whole_document=settings.knowledge_whole_document,
        search_type=search_type,  # type: ignore[arg-type]
    )
