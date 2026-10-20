"""Place an outbound Plivo call that streams to the locally running bot.

The bot must be running in Plivo mode behind a public HTTPS host::

    ngrok http 7860
    python bot.py -t plivo --proxy <ngrok-host>
    python scripts/plivo_call.py --to +91XXXXXXXXXX --answer-url https://<ngrok-host>/

Reads PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_FROM_NUMBER and MOBILE_NUMBER
from the environment / .env when the flags are omitted.
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from dotenv import load_dotenv

PLIVO_API = "https://api.plivo.com/v1/Account/{auth_id}/Call/"


def place_call(
    auth_id: str, auth_token: str, from_number: str, to_number: str, answer_url: str
) -> dict:
    """Start the call and return Plivo's JSON response (contains ``request_uuid``)."""
    response = httpx.post(
        PLIVO_API.format(auth_id=auth_id),
        auth=(auth_id, auth_token),
        json={
            "from": from_number,
            "to": to_number,
            "answer_url": answer_url,
            "answer_method": "POST",
            "hangup_url": answer_url,
            "hangup_method": "POST",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--to", default=os.getenv("MOBILE_NUMBER"), help="destination number in E.164"
    )
    parser.add_argument(
        "--from-number", default=os.getenv("PLIVO_FROM_NUMBER"), help="your Plivo number"
    )
    parser.add_argument(
        "--answer-url",
        required=True,
        help="public URL of the running bot, e.g. https://x.ngrok-free.app/",
    )
    args = parser.parse_args()

    auth_id, auth_token = os.getenv("PLIVO_AUTH_ID"), os.getenv("PLIVO_AUTH_TOKEN")
    missing = [
        n
        for n, v in (
            ("PLIVO_AUTH_ID", auth_id),
            ("PLIVO_AUTH_TOKEN", auth_token),
            ("--to/MOBILE_NUMBER", args.to),
            ("--from-number/PLIVO_FROM_NUMBER", args.from_number),
        )
        if not v
    ]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")

    result = place_call(
        auth_id, auth_token, args.from_number.lstrip("+"), args.to.lstrip("+"), args.answer_url
    )
    print(f"call requested: {result}")


if __name__ == "__main__":
    main()
