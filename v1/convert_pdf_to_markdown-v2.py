import time
from pathlib import Path
from docling.document_converter import DocumentConverter
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

def convert_pdf_to_md(pdf_path: str):

    start = time.time()

    pdf_path = Path(pdf_path)

    # ---- Input validation ----
    if not pdf_path.exists():
        print(f"[ERROR] File does not exist: {pdf_path}")
        return


    output_dir = Path("docling_md")
    output_dir.mkdir(exist_ok=True)

    output_md = output_dir / f"{pdf_path.stem}.md"

    print(f"[INFO] Input PDF: {pdf_path}")
    print("[INFO] Initializing Docling converter")

    converter = DocumentConverter()

    print("[INFO] Starting conversion...")
    result = converter.convert(str(pdf_path))

    print("[INFO] Conversion finished")
    print("[INFO] Exporting markdown")

    markdown = result.document.export_to_markdown()

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown)

    end = time.time()

    print(f"[INFO] Markdown saved to: {output_md}")
    print(f"[INFO] Total time: {round(end - start, 2)} seconds")


if __name__ == "__main__":

    pdf_file = input("Enter PDF file path: ").strip()
    convert_pdf_to_md(pdf_file)