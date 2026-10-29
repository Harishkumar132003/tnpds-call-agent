"""Sync the agent's prompts into Langfuse prompt management.

Pushes the five prompts in effect (the portal's active config version, with the
built-in text wherever nothing custom is saved) into Langfuse as versioned
prompts, labelled ``production``:

    agent/persona        the shared system prompt (voice & behaviour)
    agent/topic-check    NEW/FOLLOW_UP classifier (runs before LLM 1)
    agent/llm1-decide    LLM 1 rules: search the FAQ or reply without facts
    agent/query-rewrite  caller's messages -> one English search query
    agent/llm2-answer    LLM 2 rules: grounded answer from the sheets

Rerunning is idempotent: a prompt gets a new Langfuse version only when its
text changed, so Langfuse keeps a history that mirrors the portal's config
versions (the config version is stored on each prompt's config).

Usage:  .venv/bin/python scripts/langfuse_prompts.py
Needs the portal on :8090 and backend/.env. View under Prompts in Langfuse.
"""

from __future__ import annotations

import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langfuse import Langfuse  # noqa: E402

from voice_agent.config import load_settings  # noqa: E402

PORTAL = "http://127.0.0.1:8090"


def main() -> None:
    settings = load_settings()
    cfg = httpx.get(f"{PORTAL}/api/config", timeout=30).json()
    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )

    base_config = {
        "portal_config_version": cfg["id"],
        "model": settings.llm_model,
        "temperature": cfg.get("temperature"),
        "language": cfg.get("language") or cfg.get("default_language"),
    }
    prompts = [
        (
            "agent/persona",
            cfg.get("system_prompt") or cfg["default_prompt"],
            {"custom": cfg.get("system_prompt") is not None, "runs": "base of LLM 1 and LLM 2"},
        ),
        (
            "agent/topic-check",
            cfg.get("topic_prompt") or cfg["topic_prompt_default"],
            {"custom": cfg.get("topic_prompt") is not None, "runs": "before LLM 1, every turn"},
        ),
        (
            "agent/llm1-decide",
            cfg.get("router_prompt") or cfg["router_prompt_default"],
            {"custom": cfg.get("router_prompt") is not None, "runs": "appended to persona"},
        ),
        (
            "agent/query-rewrite",
            cfg.get("rewrite_prompt") or cfg["rewrite_prompt_default"],
            {"custom": cfg.get("rewrite_prompt") is not None, "runs": "inside the search tool"},
        ),
        (
            "agent/llm2-answer",
            cfg.get("answer_prompt") or cfg["answer_prompt_default"],
            {"custom": cfg.get("answer_prompt") is not None, "runs": "after search results"},
        ),
    ]

    # The SDK's get_prompt does not URL-encode the "/" in the name, so the
    # unchanged-check goes through the raw API with the name encoded.
    import urllib.parse

    def latest(name: str) -> dict | None:
        assert settings.langfuse_secret_key is not None
        response = httpx.get(
            f"{settings.langfuse_host}/api/public/v2/prompts/{urllib.parse.quote(name, safe='')}",
            auth=(
                settings.langfuse_public_key or "",
                settings.langfuse_secret_key.get_secret_value(),
            ),
            timeout=30,
        )
        return response.json() if response.status_code == 200 else None

    for name, text, extra in prompts:
        text = text.strip()
        current = latest(name)
        if current is not None and (current.get("prompt") or "").strip() == text:
            print(f"unchanged  {name} (version {current.get('version')})")
            continue
        created = lf.create_prompt(
            name=name,
            prompt=text,
            labels=["production"],
            tags=["voice-agent"],
            config={**base_config, **extra},
            commit_message=f"synced from portal config #{cfg['id']}",
        )
        print(f"pushed     {name} -> version {created.version}")

    lf.flush()
    print(f"\nDone. See {settings.langfuse_host} -> Prompts (portal config #{cfg['id']}).")


if __name__ == "__main__":
    main()
