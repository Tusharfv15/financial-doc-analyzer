"""
Stage 1 v2: PDF Preprocessing + OCR + Table Conversion
--------------------------------------------------------
1. Renders each page as PNG   -> layer-1-raw-md/<stem>/pages/
2. Runs Dolphin OCR            -> layer-1-raw-md/<stem>/markdown/<stem>.md
3. Injects [PAGE_START_N]      -> replaces Dolphin's --- separators in-place
4. Cleans raw markdown         -> strips figure tags in-place
5. Converts HTML tables        -> markdown tables in-place
6. Splits markdown page-wise   -> layer-1-raw-md/<stem>/markdown/page_001.md ...

Output structure per file:
    layer-1-raw-md/
    └── <stem>/
        ├── pages/
        │   ├── page_001.png
        │   └── ...
        └── markdown/
            ├── <stem>.md
            ├── page_001.md
            └── ...

Usage:
    python stage1_preprocess_v2.py             # process all PDFs in docs/
    python stage1_preprocess_v2.py myfile.pdf  # specific file
"""

import re
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
import fitz  # pymupdf
from bs4 import BeautifulSoup

INPUT_DIR  = Path("docs")
RAW_MD_DIR = Path("layer-1-raw-md")


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    pdf_name    : str
    page_images : dict[int, str]
    raw_md_path : str
    total_pages : int
    ocr_success : bool
    split_pages : dict[int, Path] = field(default_factory=dict)

    def summary(self):
        status = "✓" if self.ocr_success else "✗ OCR FAILED"
        print(f"  [{status}] {self.pdf_name}")
        print(f"  Page images : {len(self.page_images)} pages → {Path(self.raw_md_path).parent.parent / 'pages'}")
        print(f"  Raw MD      : {self.raw_md_path}")
        if self.split_pages:
            print(f"  Split pages : {len(self.split_pages)} pages → {Path(self.raw_md_path).parent}/")
        else:
            print(f"  Split pages : (none — OCR failed or no markers found)")


# ── Main entry point ───────────────────────────────────────────────────────────

def preprocess_pdf(pdf_path: str, dpi: int = 150) -> PreprocessResult:
    pdf_path = Path(pdf_path)
    stem     = pdf_path.stem

    file_dir  = RAW_MD_DIR / stem
    pages_dir = file_dir / "pages"
    md_dir    = file_dir / "markdown"

    pages_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    # ── Step 1: Render page images ─────────────────────────────────────────
    print(f"  [1/6] Rendering page images...")
    page_images = _render_pages(doc, pages_dir, dpi)
    doc.close()

    # ── Step 2: Run Dolphin OCR ────────────────────────────────────────────
    print(f"  [2/6] Running Dolphin OCR...")
    ocr_success = _run_dolphin_ocr(Path(pdf_path), file_dir)

    raw_md_path = md_dir / f"{stem}.md"
    ocr_ok      = ocr_success and raw_md_path.exists()

    # ── Step 3: Inject PAGE_START markers ─────────────────────────────────
    if ocr_ok:
        print(f"  [3/6] Injecting page markers...")
        count = _inject_page_markers(raw_md_path)
        print(f"        → {count} marker(s) injected")

    # ── Step 4: Clean raw markdown ─────────────────────────────────────────
    if ocr_ok:
        print(f"  [4/6] Cleaning raw markdown...")
        _clean_markdown(raw_md_path)

    # ── Step 5: Convert HTML tables to markdown tables ─────────────────────
    if ocr_ok:
        print(f"  [5/6] Converting HTML tables to markdown...")
        count = _convert_tables(raw_md_path)
        print(f"        → {count} table(s) converted")

    # ── Step 6: Split markdown by page ────────────────────────────────────
    split_pages: dict[int, Path] = {}
    if ocr_ok:
        print(f"  [6/6] Splitting markdown by page...")
        split_pages = _split_markdown(raw_md_path, md_dir)
        print(f"        → {len(split_pages)} page(s) written")
    else:
        print(f"  [6/6] Skipping split (OCR failed or output missing)")

    return PreprocessResult(
        pdf_name    = pdf_path.name,
        page_images = page_images,
        raw_md_path = str(raw_md_path),
        total_pages = len(page_images),
        ocr_success = ocr_ok,
        split_pages = split_pages,
    )


# ── Step 1: Render pages ───────────────────────────────────────────────────────

def _render_pages(doc: fitz.Document, pages_dir: Path, dpi: int) -> dict[int, str]:
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    images = {}
    for page_num, page in enumerate(doc, start=1):
        img_path = pages_dir / f"page_{page_num:03d}.png"
        page.get_pixmap(matrix=matrix).save(str(img_path))
        images[page_num] = str(img_path)
    return images


# ── Step 2: Dolphin OCR ────────────────────────────────────────────────────────

def _run_dolphin_ocr(pdf_path: Path, save_dir: Path) -> bool:
    try:
        subprocess.run(
            [
                sys.executable,
                "demo_page.py",
                "--model_path", "./hf_model",
                "--save_dir",   str(save_dir.resolve()),
                "--input_path", str(pdf_path.resolve()),
            ],
            cwd   = Path("Dolphin"),
            check = True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Dolphin OCR failed (return code {e.returncode})")
        return False


# ── Step 3: Inject PAGE_START markers ─────────────────────────────────────────

def _inject_page_markers(md_path: Path) -> int:
    """
    Replace Dolphin's --- page separators with [PAGE_START_N] markers.
    Operates in-place. Returns number of markers injected.
    """
    text    = md_path.read_text(encoding="utf-8")
    counter = [0]

    def replacer(match):
        counter[0] += 1
        return f"\n[PAGE_START_{counter[0] + 1}]\n"

    new_text = re.sub(r"^\s*---\s*$", replacer, text, flags=re.MULTILINE)
    md_path.write_text(new_text, encoding="utf-8")
    return counter[0]


# ── Step 4: Clean raw markdown ────────────────────────────────────────────────

def _clean_markdown(md_path: Path) -> None:
    """Remove figure tags: ![Figure](figures/...) in-place."""
    text    = md_path.read_text(encoding="utf-8")
    lines   = text.splitlines()
    cleaned = [
        line for line in lines
        if not line.strip().startswith("![Figure](figures/")
    ]
    md_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


# ── Step 5: Convert HTML tables to markdown ───────────────────────────────────

def _html_table_to_markdown(table_html: str) -> str:
    """
    Convert a single HTML table to a markdown table.
    Colspan/rowspan is flattened — cell text is preserved, span info dropped.
    """
    soup  = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if not table:
        return table_html

    rows = table.find_all("tr")
    if not rows:
        return table_html

    grid = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        grid.append([cell.get_text(strip=True) for cell in cells])

    if not grid:
        return table_html

    max_cols = max(len(row) for row in grid)
    for row in grid:
        while len(row) < max_cols:
            row.append("")

    def md_row(cells: list) -> str:
        return "| " + " | ".join(cells) + " |"

    def separator(n: int) -> str:
        return "| " + " | ".join(["---"] * n) + " |"

    lines = [md_row(grid[0]), separator(max_cols)]
    for row in grid[1:]:
        lines.append(md_row(row))

    return "\n".join(lines)


def _convert_tables(md_path: Path) -> int:
    """
    Find all HTML tables in the markdown and replace with markdown tables.
    Returns count of tables converted.
    """
    text    = md_path.read_text(encoding="utf-8")
    pattern = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    count   = 0

    for match in matches:
        md_table = _html_table_to_markdown(match)
        text     = text.replace(match, md_table, 1)
        count   += 1

    md_path.write_text(text, encoding="utf-8")
    return count


# ── Step 6: Split markdown by page ────────────────────────────────────────────

def _split_markdown(md_path: Path, out_dir: Path) -> dict[int, Path]:
    """
    Split markdown by [PAGE_START_N] markers.
    Marker appears BEFORE each page's content.
    """
    text    = md_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(\[PAGE_START_\d+\])")
    parts   = pattern.split(text)

    # parts: [preamble, marker, content, marker, content, ...]
    pairs = list(zip(parts[1::2], parts[2::2]))

    if not pairs:
        # Single page document — no separators, write entire content as page 1
        out_path = out_dir / "page_001.md"
        out_path.write_text(text.strip() + "\n", encoding="utf-8")
        return {1: out_path}

    written: dict[int, Path] = {}

    # Save preamble (content before first marker) as page_001.md
    preamble = parts[0].strip()
    if preamble:
        out_path = out_dir / "page_001.md"
        out_path.write_text(preamble + "\n", encoding="utf-8")
        written[1] = out_path

    for marker, content in pairs:
        match = re.match(r"\[PAGE_START_(\d+)\]", marker.strip())
        if not match:
            continue
        page_num = int(match.group(1))
        out_path = out_dir / f"page_{page_num:03d}.md"
        out_path.write_text(content.strip() + "\n", encoding="utf-8")
        written[page_num] = out_path

    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 2:
        pdfs = [Path(sys.argv[1])]
    else:
        if not INPUT_DIR.exists():
            print(f"[Stage 1 v2] '{INPUT_DIR}' not found. Create it and add PDFs.")
            sys.exit(1)
        pdfs = list(INPUT_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"[Stage 1 v2] No PDFs found in '{INPUT_DIR}'")
            sys.exit(1)

    print(f"[Stage 1 v2] Processing {len(pdfs)} PDF(s)\n")
    for pdf_path in pdfs:
        print(f"-> {pdf_path.name}")
        result = preprocess_pdf(str(pdf_path))
        result.summary()
        print()
    print("[Stage 1 v2] Done.")