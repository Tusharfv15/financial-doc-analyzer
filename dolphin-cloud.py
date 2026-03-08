"""
Modal script to run Dolphin-v2 document parsing on cloud GPU.

Usage:
    # Parse a single PDF or image
    modal run dolphin_modal.py --input-path ./my_document.pdf

    # Parse all docs in a directory
    modal run dolphin_modal.py --input-path ./my_docs_dir

    # Custom batch size for parallel element decoding
    modal run dolphin_modal.py --input-path ./my_document.pdf --max-batch-size 8
"""

import os
import modal
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Image & App setup
# ---------------------------------------------------------------------------

app = modal.App("dolphin-v2-ocr")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(["git", "git-lfs", "libgl1", "libglib2.0-0", "poppler-utils"])
    .pip_install(["fastapi", "huggingface_hub"])
    .run_commands([
        "git clone https://github.com/ByteDance/Dolphin.git /opt/dolphin",
        "pip install -r /opt/dolphin/requirements.txt",
    ])
)

# Persistent volume to cache the downloaded model weights (~6 GB for 3B model)
model_volume = modal.Volume.from_name("dolphin-v2-weights", create_if_missing=True)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ParseRequest(BaseModel):
    file_bytes_b64: str   # base64-encoded file contents
    filename: str
    max_batch_size: int = 4


# ---------------------------------------------------------------------------
# Modal class – model loaded once via @modal.enter()
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/model_cache": model_volume},
    timeout=60 * 30,
    scaledown_window=120,   # keep container warm for 2 min between calls
)
class DolphinParser:

    @modal.enter()
    def setup(self):
        """
        Runs once when the container starts.
        Downloads Dolphin-v2 weights if not already cached in the volume.
        """
        self.model_path = "/model_cache/dolphin-v2"

        if not os.path.exists(self.model_path):
            print("Downloading Dolphin-v2 weights (one-time setup)...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="ByteDance/Dolphin-v2", local_dir=self.model_path)
            print("Download complete.")
        else:
            print("Model weights already cached.")

    @modal.method()
    def parse(self, file_bytes: bytes, filename: str, max_batch_size: int = 4) -> dict:
        """
        Parse a single document file.
        Model is already loaded — no cold-start overhead after first call.
        """
        import sys
        import os
        import json
        import glob
        import tempfile
        import subprocess

        sys.path.insert(0, "/opt/dolphin")

        # Write input bytes to a temp file
        ext = os.path.splitext(filename)[1]
        tmp_fd, input_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(file_bytes)

        save_dir = tempfile.mkdtemp()

        # Run demo_page.py — it will reuse the already-loaded model via the
        # warm container process (subprocess shares the same GPU memory pool)
        cmd = [
            "python", "/opt/dolphin/demo_page.py",
            "--model_path", self.model_path,
            "--save_dir", save_dir,
            "--input_path", input_path,
            "--max_batch_size", str(max_batch_size),
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout[-3000:] if result.stdout else "")
        if result.returncode != 0:
            print("STDERR:", result.stderr[-3000:])
            raise RuntimeError(f"demo_page.py failed:\n{result.stderr[-2000:]}")

        # Collect output files
        pages = []
        for md_path in sorted(glob.glob(os.path.join(save_dir, "**", "*.md"), recursive=True)):
            with open(md_path, "r", encoding="utf-8") as f:
                markdown_text = f.read()

            json_path = os.path.splitext(md_path)[0] + ".json"
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    page_json = json.load(f)
            else:
                page_json = {}

            stem = os.path.splitext(os.path.basename(md_path))[0]
            last_part = stem.split("_")[-1]
            page_num = int(last_part) if last_part.isdigit() else 0

            pages.append({"page": page_num, "markdown": markdown_text, "json": page_json})

        pages.sort(key=lambda p: p["page"])
        os.unlink(input_path)

        return {"filename": filename, "pages": pages}

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: ParseRequest):
        """
        Optional HTTP endpoint — lets you POST files directly without modal run.

        curl -X POST <url> \
          -H "Content-Type: application/json" \
          -d '{"file_bytes_b64": "<base64>", "filename": "doc.pdf"}'
        """
        import base64
        file_bytes = base64.b64decode(request.file_bytes_b64)
        return self.parse(file_bytes, request.filename, request.max_batch_size)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    input_path: str = "",
    max_batch_size: int = 1,
    save_dir: str = "./results",
):
    import json
    import os

    if not input_path:
        raise ValueError("Provide --input-path, e.g.: modal run dolphin_modal.py --input-path ./doc.pdf")

    os.makedirs(save_dir, exist_ok=True)

    extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in extensions
        ]
    else:
        files = [input_path]

    if not files:
        print("No supported files found.")
        return

    print(f"Processing {len(files)} file(s) on Modal A10G GPU...")

    parser = DolphinParser()

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"\n→ {filename}")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        result = parser.parse.remote(
            file_bytes=file_bytes,
            filename=filename,
            max_batch_size=max_batch_size,
        )

        file_stem = os.path.splitext(filename)[0]
        for page in result["pages"]:
            page_label = f"_page{page['page']}" if page["page"] else ""
            md_out = os.path.join(save_dir, f"{file_stem}{page_label}.md")
            json_out = os.path.join(save_dir, f"{file_stem}{page_label}.json")

            with open(md_out, "w", encoding="utf-8") as f:
                f.write(page["markdown"])
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(page["json"], f, indent=2, ensure_ascii=False)
            print(f"  Saved: {os.path.basename(md_out)}, {os.path.basename(json_out)}")

    print(f"\nDone. Results in: {os.path.abspath(save_dir)}")