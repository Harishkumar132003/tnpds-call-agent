"""Point the dev Plivo application at the local bot (ngrok) or the server.

    uv run python scripts/plivo_point.py status          # where the number rings now
    uv run python scripts/plivo_point.py local           # https://<current ngrok host>/
    uv run python scripts/plivo_point.py prod            # https://callagent.wizzgeeks.com/
    uv run python scripts/plivo_point.py local --host abcd.ngrok-free.app
    uv run python scripts/plivo_point.py url https://other-host/

Reads PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN from .env. The application is chosen by
name (PLIVO_APP_NAME, default "TNPDS_Custom_Development"); the production
host comes from PLIVO_PROD_HOST (default callagent.wizzgeeks.com). An
application whose answer URL points at apps.sarvam.ai is never changed: that
is the live Sarvam-hosted app on the other number.

The local target is read from ngrok's local API (http://127.0.0.1:4040), so
run `ngrok http 7860` first or pass --host.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

PLIVO = "https://api.plivo.com/v1/Account/{auth_id}"
NGROK_API = "http://127.0.0.1:4040/api/tunnels"
DEFAULT_APP = "TNPDS_Custom_Development"
DEFAULT_PROD_HOST = "callagent.wizzgeeks.com"
PROTECTED = "apps.sarvam.ai"


def ngrok_host() -> str | None:
    try:
        tunnels = httpx.get(NGROK_API, timeout=3).json()["tunnels"]
    except Exception:
        return None
    for tunnel in tunnels:
        url = tunnel.get("public_url", "")
        if url.startswith("https://"):
            return url.removeprefix("https://").rstrip("/")
    return None


def _hosts_override(host: str) -> str | None:
    """The /etc/hosts address for ``host``, if any (a dev laptop may map the prod host
    to 127.0.0.1, which would make the check below lie)."""
    try:
        lines = Path("/etc/hosts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split("#", 1)[0].split()
        if len(parts) > 1 and host in parts[1:]:
            return parts[0]
    return None


def bot_answers(host: str) -> str:
    """'online' when the runner's /status answers over HTTPS at ``host``, else the reason.

    Plivo reaches the host through public DNS, so an /etc/hosts override is
    bypassed: the name is resolved with a public resolver and that address is
    used for the connection while TLS and the Host header keep the real name.
    """
    headers = {"ngrok-skip-browser-warning": "1"}
    override = _hosts_override(host)
    note = ""
    real_getaddrinfo = socket.getaddrinfo
    try:
        if override:
            import dns.resolver  # dnspython, already installed with the SDK stack

            resolver = dns.resolver.Resolver(configure=False)
            resolver.nameservers = ["1.1.1.1", "8.8.8.8"]
            address = resolver.resolve(host, "A")[0].to_text()
            note = f" (via public DNS {address}; /etc/hosts maps it to {override} here)"

            def patched(name, *args, **kwargs):
                return real_getaddrinfo(address if name == host else name, *args, **kwargs)

            socket.getaddrinfo = patched
        r = httpx.get(f"https://{host}/status", headers=headers, timeout=8, follow_redirects=True)
        return ("online" if r.status_code == 200 else f"HTTP {r.status_code}") + note
    except Exception as exc:
        return f"unreachable ({type(exc).__name__}){note}"
    finally:
        socket.getaddrinfo = real_getaddrinfo


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", choices=["status", "local", "prod", "url"])
    ap.add_argument("value", nargs="?", help="the URL for `url`")
    ap.add_argument("--host", help="host to use instead of ngrok (local) or PLIVO_PROD_HOST (prod)")
    ap.add_argument("--app", default=os.environ.get("PLIVO_APP_NAME", DEFAULT_APP))
    args = ap.parse_args()

    auth_id, token = os.environ.get("PLIVO_AUTH_ID"), os.environ.get("PLIVO_AUTH_TOKEN")
    if not auth_id or not token:
        print("PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN are needed (backend/.env)")
        return 2
    prod_host = os.environ.get("PLIVO_PROD_HOST", DEFAULT_PROD_HOST)

    with httpx.Client(
        base_url=PLIVO.format(auth_id=auth_id), auth=(auth_id, token), timeout=20
    ) as api:
        apps = api.get("/Application/").json()["objects"]
        app = next((a for a in apps if a["app_name"] == args.app), None)
        if app is None:
            print(f"no Plivo application named {args.app!r}; have: {[a['app_name'] for a in apps]}")
            return 2
        current = app.get("answer_url") or ""
        if PROTECTED in current:
            print(f"refusing: {args.app!r} points at {current} (the live Sarvam-hosted app)")
            return 2
        numbers = [
            n["number"]
            for n in api.get("/Number/").json()["objects"]
            if (n.get("application") or "").rstrip("/").endswith(app["app_id"])
        ]
        current_host = current.removeprefix("https://").removeprefix("http://").rstrip("/")
        where = (
            "prod"
            if current_host == prod_host
            else "local (ngrok)"
            if "ngrok" in current_host
            else "other"
        )
        print(f"{args.app} (number {', '.join(numbers) or 'none attached'})")
        print(f"  now: {current or '(none)'}  [{where}]")

        if args.target == "status":
            if current_host:
                print(f"  bot at that host: {bot_answers(current_host)}")
            tunnel = ngrok_host()
            print(f"  local ngrok: {tunnel or 'not running'}")
            return 0

        if args.target == "local":
            host = args.host or ngrok_host()
            if not host:
                print("  no ngrok tunnel at 127.0.0.1:4040; start `ngrok http 7860` or pass --host")
                return 2
            url = f"https://{host}/"
        elif args.target == "prod":
            url = f"https://{args.host or prod_host}/"
        else:
            if not args.value or not args.value.startswith("http"):
                print("  `url` needs a full https://... URL")
                return 2
            url = args.value if args.value.endswith("/") else args.value + "/"

        if url == current:
            print(f"  already pointing at {url}")
        else:
            r = api.post(
                f"/Application/{app['app_id']}/",
                json={"answer_url": url, "answer_method": "POST", "hangup_url": url},
            )
            if r.status_code not in (200, 202):
                print(f"  Plivo refused the update: HTTP {r.status_code} {r.text[:200]}")
                return 1
            # Plivo applies the change a moment later; read back until it shows
            after = None
            for _ in range(5):
                time.sleep(1)
                after = api.get(f"/Application/{app['app_id']}/").json().get("answer_url")
                if after == url:
                    break
            if after == url:
                print(f"  now: {after}  [{args.target}]")
            else:
                print(f"  set to {url}; Plivo still reports {after} (run `status` in a moment)")
        host = url.removeprefix("https://").rstrip("/")
        print(f"  bot at that host: {bot_answers(host)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
