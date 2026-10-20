from loguru import logger

from voice_agent.logging_config import REDACTED, SecretRedactor, configure_logging


def test_secrets_are_redacted_from_messages():
    captured: list[str] = []
    configure_logging("INFO", secrets=["sk-very-secret", ""])
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="INFO")
    try:
        logger.error("headers: {'api-subscription-key': 'sk-very-secret'} status 403")
    finally:
        logger.remove(sink_id)
    assert captured, "message was not logged"
    assert "sk-very-secret" not in captured[0]
    assert REDACTED in captured[0]


def test_redactor_handles_multiple_and_overlapping_secrets():
    record = {"message": "a=abc123 b=abc1234 c=zzz"}
    SecretRedactor(["abc1234", "abc123", "", "zzz"])(record)
    assert record["message"] == f"a={REDACTED} b={REDACTED} c={REDACTED}"
