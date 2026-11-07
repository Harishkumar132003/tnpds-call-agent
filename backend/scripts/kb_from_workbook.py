"""Build knowledge-base documents from the TNPDS extraction workbook and load them into Qdrant.

One document per ``pb*`` sheet: the topic title and line, every agent answer, the
call-forward scenarios and the example caller questions. Each document is stored as a
single chunk (the whole-document mode the agent's search was built for) with its
embedding and its BM25 sparse vector. The collection is rebuilt on every run, so a
sheet removed from the workbook disappears from Qdrant too.

Usage (from backend/):
    .venv/bin/python scripts/kb_from_workbook.py WORKBOOK.xlsx \
        [--collection NAME] [--out DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import load_workbook  # noqa: E402

from voice_agent.config import Settings  # noqa: E402
from voice_agent.knowledge.chunk import Chunk  # noqa: E402
from voice_agent.knowledge.embeddings import OpenAIEmbedder  # noqa: E402
from voice_agent.knowledge.store import KnowledgeStore  # noqa: E402

DOC_NAMESPACE = uuid.UUID("7b1e2c4a-3f5d-4e6b-9a8c-1d2e3f4a5b6c")
SHEET = re.compile(r"^pb\d+(?:-\d+)?$")


def sheet_key(name: str) -> tuple[int, int]:
    m = re.match(r"pb(\d+)(?:-(\d+))?$", name)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (10**9, 0)


def build_document(ws) -> tuple[str, str] | None:
    """Return (title, text) for one sheet, or None when the sheet has no topic row."""
    topic_title = topic_line = None
    answers: list[tuple[str, str]] = []
    forwards: list[str] = []
    questions: list[str] = []
    for section, item, content in ws.iter_rows(min_row=2, values_only=True):
        if section is None:
            continue
        content = "" if content is None else str(content).strip()
        if section == "Topic":
            topic_title, topic_line = str(item).strip(), content
        elif section == "Agent answer":
            answers.append((str(item).strip(), content))
        elif section == "Call forward":
            forwards.append(content)
        elif section == "Sample question":
            questions.append(content)
    if not topic_title:
        return None
    lines = [f"# {topic_title}", "", topic_line, ""]
    if answers:
        lines.append("## What to tell the caller")
        for item, content in answers:
            lines.append(f"{item}: {content}")
            lines.append("")
    if forwards:
        lines.append("## Forward to a human agent when")
        for f in forwards:
            lines.append(f"- {f}")
        lines.append("")
    if questions:
        lines.append("## Example caller questions")
        for q in questions:
            lines.append(f"- {q}")
        lines.append("")
    return topic_title, "\n".join(lines).rstrip() + "\n"


async def load(docs: list[tuple[str, str, str]], collection: str) -> int:
    settings = Settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not set (needed for embeddings)")
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        base_url=settings.embedding_base_url,
    )
    store = KnowledgeStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        collection=collection,
        vector_size=embedder.dimensions,
    )
    try:
        await store.recreate_collection()
        for start in range(0, len(docs), 32):
            batch = docs[start : start + 32]
            vectors = await embedder.embed([text for _, _, text in batch])
            for (sheet, title, text), vector in zip(batch, vectors, strict=True):
                document_id = str(uuid.uuid5(DOC_NAMESPACE, sheet))
                await store.upsert(document_id, f"{sheet} {title}", [Chunk(0, text)], [vector])
            print(f"  loaded {start + len(batch)}/{len(docs)}")
        return await store.point_count()
    finally:
        await store.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("workbook")
    ap.add_argument("--collection", default="voice_agent_knowledge_v2")
    ap.add_argument("--out", default="kb_docs", help="directory for the generated .md files")
    ap.add_argument("--dry-run", action="store_true", help="write the files, skip Qdrant")
    args = ap.parse_args()

    wb = load_workbook(args.workbook, read_only=True)
    names = sorted((n for n in wb.sheetnames if SHEET.match(n)), key=sheet_key)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs: list[tuple[str, str, str]] = []
    for name in names:
        built = build_document(wb[name])
        if built is None:
            print(f"  skipping {name}: no Topic row")
            continue
        title, text = built
        (out / f"{name}.md").write_text(text, encoding="utf-8")
        docs.append((name, title, text))
    chars = [len(t) for _, _, t in docs]
    print(
        f"{len(docs)} documents written to {out}/ ({min(chars)}-{max(chars)} chars, "
        f"{sum(chars) // len(chars)} avg)"
    )
    if args.dry_run:
        return
    count = asyncio.run(load(docs, args.collection))
    print(f"collection {args.collection!r} now holds {count} points")


if __name__ == "__main__":
    main()
