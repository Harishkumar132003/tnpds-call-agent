"""Register a Langfuse price for every model on the portal's Models page.

The list comes from the catalogue (voice_agent/models.py), so a model added
there is priced here with no other change. Prices are Sarvam's INR list
prices; Langfuse prints a "$" sign, but the numbers are rupees. OpenAI
embedding models are not priced: their cost is not tracked.

    uv run python scripts/langfuse_models.py                       # LANGFUSE_* from .env
    uv run python scripts/langfuse_models.py --host https://langfuse.wizzgeeks.com
    uv run python scripts/langfuse_models.py --update              # also fix changed prices
    uv run python scripts/langfuse_models.py --dry-run             # show, change nothing

Without --update, existing definitions are left alone and differences are
reported. Prices apply to traces ingested after they exist.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_agent.models import all_models  # noqa: E402


def definitions() -> list[dict]:
    seen: dict[str, dict] = {}
    for model in all_models():
        price = model.price
        if price is None or model.id in seen:
            continue
        body = {
            "modelName": model.id,
            "matchPattern": f"(?i)^({re.escape(model.id)})$",
            "unit": price.unit,
        }
        if price.input is not None:
            body["inputPrice"] = round(price.input, 12)
        if price.output is not None:
            body["outputPrice"] = round(price.output, 12)
        body["_source"] = price.source
        seen[model.id] = body
    return list(seen.values())


def _same(existing: dict, wanted: dict) -> bool:
    if existing.get("unit") != wanted.get("unit"):
        return False
    for key in ("inputPrice", "outputPrice"):
        a, b = existing.get(key), wanted.get(key)
        if (a is None) != (b is None):
            return False
        if a is not None and abs(float(a) - float(b)) > 1e-12:
            return False
    return True


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=os.environ.get("LANGFUSE_HOST"))
    ap.add_argument("--public-key", default=os.environ.get("LANGFUSE_PUBLIC_KEY"))
    ap.add_argument("--secret-key", default=os.environ.get("LANGFUSE_SECRET_KEY"))
    ap.add_argument("--update", action="store_true", help="replace definitions whose price changed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = definitions()
    if args.dry_run:
        for d in wanted:
            print(
                f"  {d['modelName']:<24} {d['unit']:<10} in={d.get('inputPrice')} "
                f"out={d.get('outputPrice')}  ({d['_source']})"
            )
        return 0
    if not (args.host and args.public_key and args.secret_key):
        print("need --host/--public-key/--secret-key or LANGFUSE_HOST/_PUBLIC_KEY/_SECRET_KEY")
        return 2

    auth = (args.public_key, args.secret_key)
    with httpx.Client(base_url=args.host.rstrip("/"), auth=auth, timeout=30) as client:
        health = client.get("/api/public/health").json()
        print(f"{args.host}: {health.get('status')} (v{health.get('version')})")
        existing = {
            m["modelName"]: m
            for m in client.get("/api/public/models?limit=100").json()["data"]
            if not m.get("isLangfuseManaged")
        }
        for d in wanted:
            name, body = d["modelName"], {k: v for k, v in d.items() if not k.startswith("_")}
            current = existing.get(name)
            if current and _same(current, body):
                print(f"  {name:<24} up to date")
                continue
            if current and not args.update:
                print(
                    f"  {name:<24} DIFFERS (in={current.get('inputPrice')} "
                    f"out={current.get('outputPrice')}); rerun with --update to replace"
                )
                continue
            if current:
                client.delete(f"/api/public/models/{current['id']}").raise_for_status()
            client.post("/api/public/models", json=body).raise_for_status()
            verb = "replaced" if current else "created"
            print(f"  {name:<24} {verb} ({body['unit']}, {d['_source']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
