"""
Stage 1: PDF Preprocessing + OCR
----------------------------------
1. Stamps [PAGE_START_N] marker at top-left of each page
2. Renders each page as PNG  -> layer-1-raw-md/<stem>/pages/
3. Runs Dolphin OCR           -> layer-1-raw-md/<stem>/markdown/<stem>_stamped.md
4. Cleans raw markdown        -> strips figure tags in-place
5. Splits markdown page-wise  -> layer-1-raw-md/<stem>/markdown/page_001.md ...

Output structure per file:
    layer-1-raw-md/
    └── <stem>/
        ├── <stem>_stamped.pdf
        ├── pages/
        │   ├── page_001.png
        │   └── ...
        └── markdown/
            ├── <stem>_stamped.md
            ├── page_001.md
            └── ...

Usage:
    python stage1_preprocess.py             # process all PDFs in docs/
    python stage1_preprocess.py myfile.pdf  # specific file
"""

import re
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
import fitz  # pymupdf

INPUT_DIR  = Path("docs")
RAW_MD_DIR = Path("layer-1-raw-md")


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    pdf_name    : str
    stamped_pdf : str
    page_images : dict[int, str]       # {page_number: image_path}
    raw_md_path : str
    total_pages : int
    ocr_success : bool
    split_pages : dict[int, Path] = field(default_factory=dict)  # {page_num: md_path}

    def summary(self):
        status = "✓" if self.ocr_success else "✗ OCR FAILED"
        print(f"  [{status}] {self.pdf_name}")
        print(f"  Stamped PDF : {self.stamped_pdf}")
        print(f"  Page images : {len(self.page_images)} pages → {Path(self.stamped_pdf).parent / 'pages'}")
        print(f"  Raw MD      : {self.raw_md_path}")
        if self.split_pages:
            print(f"  Split pages : {len(self.split_pages)} pages → {Path(self.raw_md_path).parent}/")
        else:
            print(f"  Split pages : (none — OCR failed or no markers found)")


# ── Main entry point ───────────────────────────────────────────────────────────

def preprocess_pdf(pdf_path: str, dpi: int = 150) -> PreprocessResult:
    pdf_path = Path(pdf_path)
    stem     = pdf_path.stem

    # Per-file folder structure
    file_dir  = RAW_MD_DIR / stem
    pages_dir = file_dir / "pages"
    md_dir    = file_dir / "markdown"

    pages_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    # ── Step 1: Stamp page markers ─────────────────────────────────────────
    print(f"  [1/5] Stamping page markers...")
    _stamp_page_markers(doc)

    # ── Step 2: Save stamped PDF + render page images ──────────────────────
    print(f"  [2/5] Rendering page images...")
    stamped_pdf_path = file_dir / f"{stem}_stamped.pdf"
    doc.save(str(stamped_pdf_path))
    page_images = _render_pages(doc, pages_dir, dpi)
    doc.close()

    # ── Step 3: Run Dolphin OCR ────────────────────────────────────────────
    print(f"  [3/5] Running Dolphin OCR...")
    # Pass file_dir as save_dir — Dolphin creates its own markdown/ subfolder inside it,
    # so the output lands at file_dir/markdown/<stem>_stamped.md
    ocr_success = _run_dolphin_ocr(stamped_pdf_path, file_dir)

    raw_md_path = md_dir / f"{stem}_stamped.md"
    ocr_ok      = ocr_success and raw_md_path.exists()

    # ── Step 4: Clean raw markdown ─────────────────────────────────────────
    if ocr_ok:
        print(f"  [4/5] Cleaning raw markdown...")
        _clean_markdown(raw_md_path)

    # ── Step 5: Split markdown by page ────────────────────────────────────
    split_pages: dict[int, Path] = {}
    if ocr_ok:
        print(f"  [5/5] Splitting markdown by page...")
        split_pages = _split_markdown(raw_md_path, md_dir)
        print(f"        → {len(split_pages)} page(s) written")
    else:
        print(f"  [5/5] Skipping split (OCR failed or output missing)")

    return PreprocessResult(
        pdf_name    = pdf_path.name,
        stamped_pdf = str(stamped_pdf_path),
        page_images = page_images,
        raw_md_path = str(raw_md_path),
        total_pages = len(page_images),
        ocr_success = ocr_ok,
        split_pages = split_pages,
    )


# ── Step 1: Stamp markers ──────────────────────────────────────────────────────

def _stamp_page_markers(doc: fitz.Document) -> None:
    for page_num, page in enumerate(doc, start=1):
        w = page.rect.width
        h = page.rect.height
        page.insert_textbox(
            fitz.Rect(w / 2 - 100, h - 20, w / 2 + 100, h - 5),
            f"[PAGE_START_{page_num}]",
            fontsize = 8,
            color    = (0.6, 0.6, 0.6),
            fontname = "helv",
            align    = fitz.TEXT_ALIGN_CENTER,
        )


# ── Step 2: Render pages ───────────────────────────────────────────────────────

def _render_pages(doc: fitz.Document, pages_dir: Path, dpi: int) -> dict[int, str]:
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    images = {}
    for page_num, page in enumerate(doc, start=1):
        img_path = pages_dir / f"page_{page_num:03d}.png"
        page.get_pixmap(matrix=matrix).save(str(img_path))
        images[page_num] = str(img_path)
    return images


# ── Step 3: Dolphin OCR ────────────────────────────────────────────────────────

def _run_dolphin_ocr(stamped_pdf_path: Path, save_dir: Path) -> bool:
    try:
        subprocess.run(
            [
                sys.executable,
                "demo_page.py",
                "--model_path", "./hf_model",
                "--save_dir",   str(save_dir.resolve()),
                "--input_path", str(stamped_pdf_path.resolve()),
            ],
            cwd   = Path("Dolphin"),
            check = True,
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Dolphin OCR failed (return code {e.returncode})")
        return False


# ── Step 4: Clean raw markdown ────────────────────────────────────────────────

def _clean_markdown(md_path: Path) -> None:
    """
    Clean raw Dolphin OCR markdown in-place.
    Currently removes figure tags: ![Figure](figures/...)
    """
    text    = md_path.read_text(encoding="utf-8")
    lines   = text.splitlines()

    cleaned = [
        line for line in lines
        if not line.strip().startswith("![Figure](figures/")
    ]

    md_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


# ── Step 5: Split markdown by page ────────────────────────────────────────────

def _split_markdown(md_path: Path, out_dir: Path) -> dict[int, Path]:
    """
    Split a Dolphin OCR markdown file by [PAGE_START_N] markers.
    Since the stamp is at the bottom of each page, the marker appears
    at the END of each page's content — so we split on the marker and
    assign content before it to that page number.
    Writes one .md file per page into out_dir/page_NNN.md.
    Returns {page_number: output_path}.
    """
    text = md_path.read_text(encoding="utf-8")

    # Split on the marker itself — each chunk ends with the marker
    pattern = re.compile(r"(\[PAGE_START_\d+\])")
    parts   = pattern.split(text)

    # parts alternates: [content, marker, content, marker, ...]
    # zip into (content, marker) pairs
    written: dict[int, Path] = {}
    pairs = list(zip(parts[0::2], parts[1::2]))

    if not pairs:
        print(f"  [WARN] No [PAGE_START_N] markers found in {md_path.name}")
        return {}

    for content, marker in pairs:
        match = re.match(r"\[PAGE_START_(\d+)\]", marker.strip())
        if not match:
            continue

        page_num = int(match.group(1))
        out_path = out_dir / f"page_{page_num:03d}.md"
        out_path.write_text(content.strip() + "\n", encoding="utf-8")
        written[page_num] = out_path

    # Handle any trailing content after the last marker (if any)
    if len(parts) % 2 == 1 and parts[-1].strip():
        last_page = max(written.keys()) + 1
        out_path  = out_dir / f"page_{last_page:03d}.md"
        out_path.write_text(parts[-1].strip() + "\n", encoding="utf-8")
        written[last_page] = out_path

    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 2:
        pdfs = [Path(sys.argv[1])]
    else:
        if not INPUT_DIR.exists():
            print(f"[Stage 1] '{INPUT_DIR}' not found. Create it and add PDFs.")
            sys.exit(1)
        pdfs = list(INPUT_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"[Stage 1] No PDFs found in '{INPUT_DIR}'")
            sys.exit(1)

    print(f"[Stage 1] Processing {len(pdfs)} PDF(s)\n")
    for pdf_path in pdfs:
        print(f"-> {pdf_path.name}")
        result = preprocess_pdf(str(pdf_path))
        result.summary()
        print()
    print("[Stage 1] Done.")