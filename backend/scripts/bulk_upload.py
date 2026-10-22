"""Upload a folder of documents to the knowledge base through a portal's API.

    uv run python scripts/bulk_upload.py "~/Documents/FAQ Full Set 25th May26"
    uv run python scripts/bulk_upload.py DIR --portal https://callagentportal.wizzgeeks.com
    uv run python scripts/bulk_upload.py DIR --exclude "001 - Home.pdf" --check

Files already indexed under the same name (status ready) are skipped, so a
rerun only fills gaps; documents left failed or needs-reupload are replaced.
Uploads go one at a time with retries. --check runs sample questions afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
DEFAULT_EXCLUDE = ["001 - Home.pdf"]  # the topic index: would win most top-1 searches
SAMPLE_QUESTIONS = [
    ("ரேஷன் கார்டில் இருந்து ஒரு உறுப்பினரை நீக்குவது எப்படி?", "Remove member"),
    ("புதிய ரேஷன் கார்டுக்கு ஆன்லைனில் எப்படி விண்ணப்பிப்பது?", "NSC-online"),
    ("ரேஷன் கடை பற்றி புகார் கொடுப்பது எப்படி?", "Complaint"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=lambda p: Path(p).expanduser())
    ap.add_argument("--portal", default="http://127.0.0.1:8090")
    ap.add_argument(
        "--exclude", action="append", default=None, help="file name to skip (repeatable)"
    )
    ap.add_argument("--check", action="store_true", help="run sample questions afterwards")
    args = ap.parse_args()
    exclude = set(args.exclude if args.exclude is not None else DEFAULT_EXCLUDE)

    files = sorted(f for f in args.folder.iterdir() if f.suffix.lower() in EXTENSIONS)
    with httpx.Client(base_url=args.portal.rstrip("/"), timeout=300) as api:
        status = api.get("/api/knowledge/status").json()
        if not (status.get("qdrant_ok") and status.get("embedding_key_set")):
            print(f"knowledge base not ready at {args.portal}: {status}")
            return 2
        existing = {d["name"]: d for d in api.get("/api/documents").json()}
        print(f"{args.portal}: {len(existing)} documents, {status.get('points')} chunks before")

        done, skipped, failed = 0, 0, []
        for i, f in enumerate(files, 1):
            if f.name in exclude:
                skipped += 1
                continue
            old = existing.get(f.name)
            if old and old["status"] == "ready":
                skipped += 1
                continue
            if old:  # failed / needs-reupload: replace it
                api.delete(f"/api/documents/{old['id']}")
            for attempt in range(3):
                try:
                    r = api.post("/api/documents", files={"file": (f.name, f.read_bytes())})
                    if r.status_code < 300:
                        done += 1
                        print(
                            f"[{i}/{len(files)}] {f.name}: {r.json().get('chunk_count')} chunk(s)"
                        )
                        break
                    error = f"HTTP {r.status_code} {r.text[:120]}"
                except httpx.HTTPError as exc:
                    error = type(exc).__name__
                time.sleep(2 * (attempt + 1))
            else:
                failed.append((f.name, error))
                print(f"[{i}/{len(files)}] {f.name}: FAILED {error}")

        status = api.get("/api/knowledge/status").json()
        print(
            print(f"\nindexed {done}, skipped {skipped}, failed {len(failed)}")
            print(f"{status.get('points')} chunks in the collection now")
        )
        for name, error in failed:
            print(f"  failed: {name}: {error}")

        if args.check:
            print("\nretrieval check (top 3):")
            for question, expect in SAMPLE_QUESTIONS:
                r = api.post(
                    "/api/knowledge/search", json={"question": question, "limit": 3}
                ).json()
                hits = [(h["document_name"], round(h["score"], 2)) for h in r["hits"]]
                ok = "ok  " if hits and expect.lower() in hits[0][0].lower() else "MISS"
                print(f"  {ok} {question}\n       -> {hits}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
