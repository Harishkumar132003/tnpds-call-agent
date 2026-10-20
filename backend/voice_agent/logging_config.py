"""Loguru configuration shared by the application entrypoints."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any

from loguru import logger

REDACTED = "***REDACTED***"


class SecretRedactor:
    """Loguru patcher that masks secret values in every log message.

    Third-party SDKs sometimes include request headers (and therefore API keys)
    in exception text that ends up in the logs; this keeps them out of the
    message. Exception tracebacks are not patched, which is why
    :func:`configure_logging` disables loguru's variable-dumping ``diagnose``.
    """

    def __init__(self, secrets: Iterable[str]) -> None:
        self._secrets = tuple(s for s in secrets if s)

    def __call__(self, record: dict[str, Any]) -> None:
        message = record["message"]
        for secret in self._secrets:
            if secret in message:
                message = message.replace(secret, REDACTED)
        record["message"] = message


def configure_logging(level: str, *, secrets: Iterable[str] = ()) -> None:
    """Replace loguru's sinks with one at ``level`` and redact ``secrets``.

    ``backtrace``/``diagnose`` are off so tracebacks never print local
    variables such as ``api_key``.
    """
    logger.remove()
    logger.configure(patcher=SecretRedactor(secrets))
    logger.add(sys.stderr, level=level, backtrace=False, diagnose=False)
