"""bot.py serves Plivo's answer webhook when the runner is not restricted to one transport."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bot import register_plivo_webhook


def test_webhook_served_without_transport_flag():
    app = FastAPI()
    assert register_plivo_webhook(app, ["--proxy", "calls.example.com", "--port", "7860"])
    r = TestClient(app).post("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/xml")
    assert "wss://calls.example.com/ws" in r.text
    assert 'bidirectional="true"' in r.text


def test_runner_keeps_its_own_route_with_transport_flag():
    app = FastAPI()
    assert not register_plivo_webhook(app, ["-t", "plivo", "--proxy", "calls.example.com"])
    assert TestClient(app).post("/").status_code in (404, 405)


def test_no_proxy_no_route():
    assert not register_plivo_webhook(FastAPI(), ["--port", "7860"])
