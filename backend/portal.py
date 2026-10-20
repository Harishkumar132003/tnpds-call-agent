"""Entrypoint for the operations portal: ``python portal.py`` then open http://localhost:8090."""

from __future__ import annotations

import uvicorn
from loguru import logger

from portal.app import create_app
from voice_agent.config import ConfigurationError, load_settings
from voice_agent.logging_config import configure_logging


def main() -> None:
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    if settings.database_url is None:
        logger.error("DATABASE_URL is not set; the portal needs the tracking database.")
        raise SystemExit(1)
    configure_logging(settings.log_level, secrets=[settings.sarvam_api_key.get_secret_value()])
    app = create_app(
        settings.database_url,
        default_prompt_language=settings.agent_language,
        agent_name=settings.agent_name,
        knowledge=settings,
    )
    uvicorn.run(app, host=settings.portal_host, port=settings.portal_port, log_level="info")


if __name__ == "__main__":
    main()
