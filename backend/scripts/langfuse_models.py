"""Register Sarvam model prices in a Langfuse project so traces get costs.

Prices are Sarvam's INR list prices; Langfuse prints them with a "$" sign but
the numbers are rupees. Idempotent: existing definitions with the same name
are left alone (delete them in Settings -> Models to re-create).

    uv run python scripts/langfuse_models.py                 # uses LANGFUSE_* from .env
    uv run python scripts/langfuse_models.py --host https://langfuse.wizzgeeks.com
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from dotenv import load_dotenv

MODELS = [
    {
        "modelName": "sarvam-105b",
        "matchPattern": "(?i)^(sarvam-105b)$",
        "unit": "TOKENS",
        "inputPrice": 0.000015,  # ₹15 per 1M input tokens
        "outputPrice": 0.00006,  # ₹60 per 1M output tokens
    },
    {
        "modelName": "bulbul:v3",
        "matchPattern": "(?i)^(bulbul:v3)$",
        "unit": "CHARACTERS",
        "outputPrice": 0.003,  # ₹30 per 10k characters
    },
    {
        "modelName": "sarvam-translate:v1",
        "matchPattern": "(?i)^(sarvam-translate:v1)$",
        "unit": "CHARACTERS",
        "inputPrice": 0.002,  # ₹2 per 1,000 characters (pay as you go)
    },
    {
        "modelName": "mayura:v1",
        "matchPattern": "(?i)^(mayura:v1)$",
        "unit": "CHARACTERS",
        "inputPrice": 0.002,  # same translate rate
    },
    {
        "modelName": "saaras:v4",
        "matchPattern": "(?i)^(saaras:v4)$",
        "unit": "SECONDS",
        "inputPrice": 0.008333,  # ₹30 per hour of audio
    },
]


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=os.environ.get("LANGFUSE_HOST"))
    ap.add_argument("--public-key", default=os.environ.get("LANGFUSE_PUBLIC_KEY"))
    ap.add_argument("--secret-key", default=os.environ.get("LANGFUSE_SECRET_KEY"))
    args = ap.parse_args()
    if not (args.host and args.public_key and args.secret_key):
        print("need --host/--public-key/--secret-key or LANGFUSE_HOST/_PUBLIC_KEY/_SECRET_KEY")
        return 2

    auth = (args.public_key, args.secret_key)
    with httpx.Client(base_url=args.host.rstrip("/"), auth=auth, timeout=30) as client:
        health = client.get("/api/public/health")
        print(f"{args.host}: {health.json().get('status')} (v{health.json().get('version')})")
        existing = {
            m["modelName"] for m in client.get("/api/public/models?limit=100").json()["data"]
        }
        for model in MODELS:
            if model["modelName"] in existing:
                print(f"  {model['modelName']:<12} already defined")
                continue
            r = client.post("/api/public/models", json=model)
            r.raise_for_status()
            print(f"  {model['modelName']:<12} created ({model['unit']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
