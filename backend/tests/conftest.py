"""Shared fixtures. External services are never contacted in the unit tests."""

from __future__ import annotations

import pytest

from voice_agent.config import Settings

FAKE_SARVAM_KEY = "sk_test_not_a_real_key"


@pytest.fixture
def make_settings():
    """Build ``Settings`` from explicit values only, ignoring ``.env`` and the real env."""

    def _make(**overrides) -> Settings:
        values = {"sarvam_api_key": FAKE_SARVAM_KEY}
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return _make


@pytest.fixture
def settings(make_settings) -> Settings:
    return make_settings()
