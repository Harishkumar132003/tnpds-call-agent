"""Export every sheet of a workbook as its own PDF, laid out for text chunking.

Each sheet becomes one PDF: a heading (the sheet's title cell or its name), a
source line, then the sheet's non-empty rows as paragraphs in reading order.
Cells in a row are joined with " — ", blank rows become paragraph breaks and
lines ending in ":" are set bold as step headings. Text stays selectable, so
``voice_agent.knowledge.extract`` recovers it verbatim for chunking.

Usage::

    python scripts/xlsx_to_pdfs.py "~/Downloads/FAQ.xlsx" --out "~/Documents"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

import openpyxl
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

FONT_DIR = Path("/usr/share/fonts/truetype/noto")
FONTS = {
    "Body": FONT_DIR / "NotoSans-Regular.ttf",
    "Body-Bold": FONT_DIR / "NotoSans-Bold.ttf",
    "Tamil": FONT_DIR / "NotoSansTamil-Regular.ttf",
}
_TAMIL = re.compile(r"[஀-௿]")


def register_fonts() -> None:
    for name, path in FONTS.items():
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))


def styles() -> dict[str, ParagraphStyle]:
    body_font = "Body" if "Body" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold_font = (
        "Body-Bold" if "Body-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
    )
    return {
        "title": ParagraphStyle("title", fontName=bold_font, fontSize=16, leading=20, spaceAfter=4),
        "source": ParagraphStyle(
            "source",
            fontName=body_font,
            fontSize=8.5,
            leading=11,
            textColor="#666666",
            spaceAfter=10,
        ),
        "body": ParagraphStyle("body", fontName=body_font, fontSize=10.5, leading=15, spaceAfter=4),
        "heading": ParagraphStyle(
            "heading", fontName=bold_font, fontSize=10.5, leading=15, spaceBefore=6, spaceAfter=3
        ),
    }


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s().,&-]+", " ", name, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()[:120] or "sheet"


def sheet_lines(ws) -> tuple[str | None, list[str]]:
    """Return (title, lines). The first row that reads ``Home | <title>`` names the sheet."""
    title = None
    lines: list[str] = []
    for row in ws.iter_rows(values_only=True):
        cells = [str(c).strip() for c in row if c is not None and str(c).strip() != ""]
        if not cells:
            lines.append("")
            continue
        if title is None and len(cells) >= 2 and cells[0].lower() == "home":
            title = cells[1]
            continue
        if title is None and len(cells) == 1 and cells[0].lower() == "home":
            continue
        if len(cells) == 1 and cells[0].lower() in {"query", "complaint", "query ", "complaint "}:
            continue
        lines.append(" — ".join(cells))
    # collapse runs of blank lines
    out: list[str] = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return title, out


def render_pdf(path: Path, title: str, source: str, lines: list[str]) -> None:
    st = styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="xlsx_to_pdfs",
    )
    story = [Paragraph(escape(title), st["title"]), Paragraph(escape(source), st["source"])]
    for line in lines:
        if line == "":
            story.append(Spacer(1, 6))
            continue
        text = escape(line).replace("\n", "<br/>")
        if _TAMIL.search(line) and "Tamil" in pdfmetrics.getRegisteredFontNames():
            text = f'<font name="Tamil">{text}</font>'
        style = st["heading"] if line.endswith(":") and len(line) < 120 else st["body"]
        story.append(Paragraph(text, style))
    doc.build(story)


def convert(workbook: Path, out_root: Path) -> list[Path]:
    register_fonts()
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    folder = out_root / workbook.stem
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, ws in enumerate(wb.worksheets, start=1):
        title, lines = sheet_lines(ws)
        if not lines and not title:
            continue
        heading = title or ws.title.strip()
        source = f"Source: {workbook.name}, sheet '{ws.title.strip()}'"
        path = folder / f"{index:03d} - {safe_filename(ws.title)}.pdf"
        render_pdf(path, heading, source, lines)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workbook", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "Documents",
        help="parent folder; a sub-folder named after the workbook is created",
    )
    args = parser.parse_args()
    written = convert(args.workbook.expanduser(), args.out.expanduser())
    print(f"wrote {len(written)} PDFs to {written[0].parent if written else args.out}")


if __name__ == "__main__":
    main()
