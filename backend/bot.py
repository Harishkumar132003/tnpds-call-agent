"""Entrypoint for the Pipecat development runner.

Run ``python bot.py`` and open http://localhost:7860/client in a browser.
"""

from __future__ import annotations

import os
import sys

from loguru import logger
from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.runner.utils import create_transport

from voice_agent.config import ConfigurationError, Settings, load_settings
from voice_agent.logging_config import configure_logging
from voice_agent.services import validate_service_configuration
from voice_agent.session import run_voice_session
from voice_agent.tracking.recorder import CallInfo
from voice_agent.transports import TRANSPORT_PARAMS, audio_rates_for

# Loaded once in main(); the runner calls bot() from this same module.
_settings: Settings | None = None
_session_logging_configured = False


async def bot(runner_args: RunnerArguments) -> None:
    """Called by the runner once per connected client."""
    settings = _settings or load_settings()
    _configure_session_logging(settings)
    try:
        transport = await create_transport(runner_args, TRANSPORT_PARAMS)
        await run_voice_session(
            transport,
            settings,
            handle_sigint=runner_args.handle_sigint,
            audio_rates=audio_rates_for(runner_args),
            call_info=_call_info(runner_args, settings),
        )
    except Exception:
        logger.exception("Voice session failed")
        await _close_client_connection(runner_args)
        raise


def _call_info(runner_args: RunnerArguments, settings: Settings) -> CallInfo:
    """Describe the session for call tracking from what the runner tells us."""
    if isinstance(runner_args, WebSocketRunnerArguments):
        call = runner_args.call_data
        ours = (settings.plivo_from_number or "").lstrip("+")
        to_number = (call.to_number if call else None) or None
        from_number = (call.from_number if call else None) or None
        direction = None
        if ours and to_number and to_number.lstrip("+") == ours:
            direction = "inbound"
        elif ours and from_number and from_number.lstrip("+") == ours:
            direction = "outbound"
        return CallInfo(
            transport=runner_args.transport_type or "telephony",
            language=settings.agent_language,
            session_id=runner_args.session_id,
            direction=direction,
            from_number=from_number,
            to_number=to_number,
            call_id=call.call_id if call else None,
            stream_id=call.stream_id if call else None,
        )
    return CallInfo(
        transport="webrtc",
        language=settings.agent_language,
        session_id=runner_args.session_id,
        direction="web",
    )


def _configure_session_logging(settings: Settings) -> None:
    """Re-apply our logging once: the runner resets loguru to DEBUG after main()."""
    global _session_logging_configured
    if not _session_logging_configured:
        configure_logging(settings.log_level, secrets=_secret_values(settings))
        _session_logging_configured = True


async def _close_client_connection(runner_args: RunnerArguments) -> None:
    """Close the WebRTC peer so the browser does not wait forever on 'connecting'."""
    connection = getattr(runner_args, "webrtc_connection", None)
    if connection is not None:
        await connection.disconnect()


def runner_args_from_settings(argv: list[str], settings: Settings) -> list[str]:
    """Fill in ``--proxy`` / ``-t`` from PUBLIC_HOST / BOT_TRANSPORT when not given on the CLI.

    Explicit flags always win, so ``python bot.py -t webrtc`` still works locally
    while the deployed container needs nothing but the .env file.
    """
    args = list(argv)
    if settings.public_host and "--proxy" not in args and "-x" not in args:
        host = settings.public_host.removeprefix("https://").removeprefix("http://").strip("/")
        args += ["--proxy", host]
    if settings.bot_transport and "-t" not in args and "--transport" not in args:
        args += ["-t", settings.bot_transport]
    return args


def _drop_blank_env_vars() -> None:
    """The runner exports every ``.env`` line, so ``OPENAI_BASE_URL=`` becomes an empty
    environment variable that SDKs then treat as a real (broken) base URL."""
    for key in [k for k, v in os.environ.items() if v == "" and k.isupper()]:
        del os.environ[key]


def _secret_values(settings: Settings) -> list[str]:
    secrets = [settings.sarvam_api_key.get_secret_value()]
    for secret in (settings.openai_api_key, settings.plivo_auth_token):
        if secret is not None:
            secrets.append(secret.get_secret_value())
    return secrets


def main() -> None:
    """Validate configuration, then hand over to the Pipecat runner's CLI."""
    # Importing the runner loads .env into the environment (override=True), so
    # settings must be read after it to match what the runner itself sees.
    from pipecat.runner.run import main as run_pipecat

    _drop_blank_env_vars()
    global _settings
    try:
        _settings = load_settings()
        validate_service_configuration(_settings)
    except ConfigurationError as exc:
        logger.error(str(exc))
        logger.error("Copy .env.example to .env and fill in the missing values.")
        raise SystemExit(1) from exc
    configure_logging(_settings.log_level, secrets=_secret_values(_settings))
    sys.argv[1:] = runner_args_from_settings(sys.argv[1:], _settings)
    run_pipecat()


if __name__ == "__main__":
    main()
