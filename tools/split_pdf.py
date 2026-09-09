"""Split the book PDF (found automatically in the repo root) into one PDF + one text file per chapter, using book.yaml.

Repo root = C:/Users/sislam27/Work/Climate Dynamics PHD/Sea Ice Image Processing with MATLAB
The book is the single *.pdf sitting in that root ("Sea ice image processing with MATLAB® by Skjetne ... .pdf");
book.yaml → book.pdf is "auto" (pick the largest *.pdf in the root, ignoring chapters/) or an explicit file name.

Usage (from repo root):
    python tools/split_pdf.py            # auto-detect offset, write chapters/chNN.pdf and chapters/chNN.txt
    python tools/split_pdf.py --offset 18   # force offset (pdf_page = printed_page + offset)
    python tools/split_pdf.py --dry-run     # only print the detected mapping

Offset detection: the book prints page numbers on each page. We look for the page whose text
starts Chapter 2 ("Digital Image Processing Preliminaries") and printed page 11, and derive
offset = pdf_index - printed_page. We also write chapters/_page_map.json for inspection.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

try:
    import fitz  # pymupdf
except ImportError:  # pragma: no cover
    fitz = None
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]


def load_book() -> dict:
    with open(ROOT / "book.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def locate_book_pdf(book: dict) -> Path | None:
    """Return the book PDF path. 'auto' → largest *.pdf directly in the repo root (never inside chapters/)."""
    spec = str(book["book"].get("pdf", "auto") or "auto")
    if spec != "auto":
        cand = ROOT / spec
        return cand if cand.exists() else None
    pdfs = [p for p in ROOT.glob("*.pdf") if p.is_file()]
    if not pdfs:
        return None
    return max(pdfs, key=lambda p: p.stat().st_size)


def _clean_text(t: str) -> str:
    """Drop NUL bytes that PDF text extraction sometimes emits (they make grep treat the .txt as binary)."""
    return t.replace("\x00", "")


def page_texts(pdf_path: Path) -> list[str]:
    if fitz is not None:
        doc = fitz.open(pdf_path)
        return [_clean_text(p.get_text("text")) for p in doc]
    reader = PdfReader(str(pdf_path))
    return [_clean_text(p.extract_text() or "") for p in reader.pages]


def _has_printed_number(text: str, number: int, edge_lines: int = 3) -> bool:
    """True if `number` appears as a standalone line among the first/last `edge_lines` non-empty lines.

    In this book the running head puts the page number at the top of body pages and chapter openers put it
    at the bottom; the table of contents (which also mentions every chapter title) carries roman numerals.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    edge = lines[:edge_lines] + lines[-edge_lines:]
    return any(ln == str(number) for ln in edge)


def detect_offset(texts: list[str], chapters: list[dict]) -> int | None:
    """Find offset such that pdf_index (0-based) = printed_page + offset - 1.

    Strategy: for each porting chapter, every page that contains the chapter title (case-insensitive,
    whitespace-collapsed) *and* looks like a chapter opener (a line 'N' or 'Chapter N') is a candidate.
    A candidate whose page also carries the chapter's printed start page number as a standalone line at the
    top or bottom of the page text scores 10; one without scores 1 (ToC / cross-reference pages typically lack
    it). The offset with the highest total score wins; ties go to the smallest |offset|.
    """
    from collections import Counter

    scores: Counter[int] = Counter()
    for ch in chapters:
        if not ch.get("do_port", False) or ch["number"] > 9:
            continue
        title = re.sub(r"\s+", " ", ch["title"]).lower()
        key = " ".join(title.split(" ")[:4])
        for i, t in enumerate(texts):
            tt = re.sub(r"\s+", " ", t).lower()
            if key not in tt or not re.search(rf"(chapter\s+{ch['number']}\b|^\s*{ch['number']}\s)", tt):
                continue
            # skip the table of contents: it lists many chapters on one page
            if tt.count("chapter") > 3:
                continue
            offset = i - (ch["printed_page_start"] - 1)
            scores[offset] += 10 if _has_printed_number(t, ch["printed_page_start"]) else 1
    if not scores:
        return None
    best = max(scores.values())
    return min((o for o, sc in scores.items() if sc == best), key=abs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=None, help="pdf_index0 = printed_page - 1 + offset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    book = load_book()
    pdf_path = locate_book_pdf(book)
    if pdf_path is None:
        print(f"ERROR: no book PDF found in {ROOT}. The book PDF must sit directly in the repo root, "
              "or set book.pdf in book.yaml to its file name.")
        return 1
    print(f"Book PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)")
    out_dir = ROOT / "chapters"
    out_dir.mkdir(exist_ok=True)

    texts = page_texts(pdf_path)
    n_pages = len(texts)
    offset = args.offset if args.offset is not None else detect_offset(texts, book["chapters"])
    if offset is None:
        print("Could not auto-detect the page offset. Open the book PDF, find the PDF page index of printed page 11 "
              "(start of Chapter 2) and rerun with --offset <pdf_index_of_page_11 - 11 + 1>.")
        print(f"(book PDF used: {pdf_path.name})")
        return 2
    print(f"PDF has {n_pages} pages; using offset={offset} (pdf 0-based index = printed_page - 1 + offset)")

    page_map = {}
    reader = PdfReader(str(pdf_path))
    for ch in book["chapters"]:
        start = ch["printed_page_start"] - 1 + offset
        end = ch["printed_page_end"] - 1 + offset  # inclusive
        start = max(0, start)
        end = min(n_pages - 1, end)
        page_map[ch["id"]] = {"title": ch["title"], "pdf_index_start": start, "pdf_index_end": end,
                              "printed": [ch["printed_page_start"], ch["printed_page_end"]]}
        first_line = texts[start].strip().splitlines()[:3] if texts[start].strip() else []
        print(f"  {ch['id']}: pdf pages {start + 1}-{end + 1}  | first lines: {first_line}")
        if args.dry_run:
            continue
        writer = PdfWriter()
        for i in range(start, end + 1):
            writer.add_page(reader.pages[i])
        with open(out_dir / f"{ch['id']}.pdf", "wb") as f:
            writer.write(f)
        with open(out_dir / f"{ch['id']}.txt", "w", encoding="utf-8") as f:
            for i in range(start, end + 1):
                f.write(f"\n\n===== PDF page {i + 1} (printed page {i + 1 - offset}) =====\n")
                f.write(texts[i])
    with open(out_dir / "_page_map.json", "w", encoding="utf-8") as f:
        json.dump({"offset": offset, "n_pages": n_pages, "chapters": page_map}, f, indent=2)
    print(f"Wrote {out_dir}/chNN.pdf, chNN.txt and _page_map.json. "
          f"CHECK: chapters/ch02.txt must begin with 'Digital Image Processing Preliminaries' (printed page 11).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
