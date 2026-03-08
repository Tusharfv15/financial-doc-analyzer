"""
Modal script to run the Financial Document Insights Engine on the cloud.

Usage:
    # Run with a PDF
    modal run narrator_modal.py --input-path ./statement.pdf --query "What is my total spend?"

    # Run with a markdown file
    modal run narrator_modal.py --input-path ./statement.md --query "What is my total spend?"
"""

import os
import json
import time
import modal
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI

MODEL   = "gpt-5-mini-2025-08-07"
DIVIDER = "─" * 60

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# ---------------------------------------------------------------------------
# Image & App setup
# ---------------------------------------------------------------------------

app = modal.App("financial-insights-engine")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libgl1",
        "libglib2.0-0",
        "poppler-utils",
    ])
    .pip_install([
        "openai",
        "docling",
        "pydantic",
        "fastapi",
        "rapidocr-onnxruntime",
        "opencv-python-headless",
    ])
    .add_local_python_source("schema_planner")
    .add_local_python_source("extractor")
    .add_local_python_source("router")
    .add_local_python_source("code_generator")
    .add_local_python_source("code_executor")
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class NarratorResponse(BaseModel):
    answer  : str = Field(description="Clear, concise natural language answer to the user's query")
    summary : str = Field(description="One-line summary of the key insight e.g. 'Total spend: PHP 45,000'")


data_volume = modal.Volume.from_name("financial-insights-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=60 * 10,
    scaledown_window=120,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("openai-secret")],
)
class FinancialInsightsEngine:

    @modal.enter()
    def setup(self):
        import os
        from docling.document_converter import DocumentConverter

        # Point Docling model cache to the persistent volume
        os.environ["DOCLING_CACHE_DIR"] = "/data/docling-cache"
        os.makedirs("/data/docling-cache", exist_ok=True)
        os.makedirs("/data/results", exist_ok=True)

        print("Initializing OpenAI client...")
        self.client = OpenAI()
        print("Initializing Docling converter...")
        self.converter = DocumentConverter()
        print("Ready.")

    def convert_to_md(self, input_path: str) -> str:
        start      = time.time()
        input_path = Path(input_path) # type: ignore

        if not input_path.exists(): # type: ignore
            raise FileNotFoundError(f"File does not exist: {input_path}")

        ext = input_path.suffix.lower() # type: ignore
        if ext not in {".pdf", ".csv", ".xlsx"}:
            raise ValueError(f"Unsupported file type: {ext}. Provide a PDF, CSV, or XLSX.")

        output_dir = Path("/data/results")
        output_dir.mkdir(exist_ok=True)
        output_md = output_dir / f"{input_path.stem}.md" # type: ignore

        print(f"[INFO] Input file : {input_path}")
        print(f"[INFO] File type  : {ext}")
        print("[INFO] Starting conversion...")
        result = self.converter.convert(input_path)

        print("[INFO] Conversion finished — exporting markdown")
        markdown = result.document.export_to_markdown()

        with open(output_md, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"[INFO] Markdown saved to: {output_md}")
        print(f"[INFO] Total time: {round(time.time() - start, 2)}s\n")
        return str(output_md)

    def narrator_node(
        self,
        query        : str,
        statement    : dict,
        transactions : list[dict],
        result       : str | None = None,
    ) -> NarratorResponse:
        data_context = (
            f"Computed result:\n{result}"
            if result else
            f"Extracted records ({len(transactions)}):\n{json.dumps(transactions, indent=2)}"
        )

        response = self.client.responses.parse(
            model  = MODEL,
            input  = [
                {
                    "role"    : "system",
                    "content" : (
                        "You are a helpful financial document assistant. "
                        "You will be given a user query, extracted financial data, and a computed result. "
                        "Your job is to narrate the answer clearly and naturally.\n\n"
                        "Rules:\n"
                        "- Be concise and direct — answer the query first, then provide context.\n"
                        "- Always mention the currency if present in the data.\n"
                        "- If the result is a list of transactions, present them in a readable format.\n"
                        "- If the result is a number, round to 2 decimal places.\n"
                        "- Highlight any notable observations e.g. a payment included in the average.\n"
                        "- Do not mention technical details like 'extracted data' or 'computed result'.\n"
                        "- Keep the tone professional but friendly."
                    )
                },
                {
                    "role"    : "user",
                    "content" : (
                        f"Query: {query}\n\n"
                        f"Statement summary:\n{json.dumps(statement, indent=2)}\n\n"
                        f"{data_context}"
                    )
                }
            ],
            text_format = NarratorResponse,
        )

        return response.output_parsed  # type: ignore

    @modal.method()
    def run_pipeline(self, input_bytes: bytes, filename: str, query: str) -> dict:
        import os
        import tempfile
        from schema_planner import schema_planner_node
        from extractor import extractor_node
        from router import router_node
        from code_generator import code_generator_node
        from code_executor import executor_node, MAX_RETRIES

        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

        # Write input file to temp location
        ext = os.path.splitext(filename)[1].lower()
        tmp_fd, input_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(input_bytes)

        # Resolve markdown
        if ext in {".pdf", ".csv", ".xlsx"}:
            md_path  = self.convert_to_md(input_path)
            markdown = open(md_path, encoding="utf-8").read()
        elif ext == ".md":
            markdown = open(input_path, encoding="utf-8").read()
        else:
            raise ValueError(f"Unsupported file type: {ext}. Provide a PDF, CSV, XLSX, or Markdown file.")

        os.unlink(input_path)

        # Step 1 – Schema Planner
        print(f"[1/6] Schema Planner")
        print(f"      {DIVIDER}")
        schema = schema_planner_node(query, markdown)
        print(f"      Fields planned : {len(schema.fields)}")
        for f in schema.fields:
            print(f"        • {f.name} ({f.type}) — {f.description}")
        print(f"      Filter hint    : {schema.filter_hint}")
        print(f"      Compute hint   : {schema.computation_hint}")
        print(f"      Doc context    : {schema.document_context}")
        print()

        # Step 2 – Extractor
        print(f"[2/6] Extractor")
        print(f"      {DIVIDER}")
        statement, transactions = extractor_node(query, schema, markdown)
        print(f"      Transactions extracted : {len(transactions)}")
        print(f"      Statement fields       : {list(statement.keys())}")
        print(f"      Statement values:")
        for k, v in statement.items():
            print(f"        • {k} : {v}")
        print(f"      Transactions:")
        for i, tx in enumerate(transactions, 1):
            print(f"        [{i:02d}] {json.dumps(tx)}")
        print()

        # Step 3 – Router
        print(f"[3/6] Router")
        print(f"      {DIVIDER}")
        decision = router_node(query, statement, transactions)
        print(f"      Needs computation : {decision.needs_computation}")
        print(f"      Reason            : {decision.reason}")
        print()

        exec_result = None

        if decision.needs_computation:
            # Step 4 – Code Generator
            print(f"[4/6] Code Generator")
            print(f"      {DIVIDER}")
            generated = code_generator_node(query, schema, statement, transactions)
            print(f"      Reasoning : {generated.reasoning}")
            print(f"      Generated code:")
            for line in generated.code.splitlines():
                print(f"        {line}")
            print()

            # Step 5 – Executor
            print(f"[5/6] Executor")
            print(f"      {DIVIDER}")
            exec_result = executor_node(query, schema, statement, transactions, generated)
            if exec_result.success:
                print(f"      ✓ Execution successful")
                print(f"      Raw result : {exec_result.result}")
            else:
                print(f"      ✗ Execution failed after {MAX_RETRIES} attempts")
                print(f"      Error : {exec_result.error}")
            print()
        else:
            print(f"[4/6] Code Generator  — SKIPPED (not needed)")
            print(f"[5/6] Executor        — SKIPPED (not needed)")
            print()

        # Step 6 – Narrator
        print(f"[6/6] Narrator")
        print(f"      {DIVIDER}")
        narrator = self.narrator_node(
            query        = query,
            statement    = statement,
            transactions = transactions,
            result       = exec_result.result if exec_result and exec_result.success else None,
        )

        return {
            "summary"           : narrator.summary,
            "answer"            : narrator.answer,
            "needs_computation" : decision.needs_computation,
            "exec_success"      : exec_result.success if exec_result else None,
            "transactions_count": len(transactions),
        }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    input_path : str = "",
    query      : str = "",
):
    if not input_path or not query:
        raise ValueError(
            'Provide both --input-path and --query\n'
            'e.g.: modal run narrator_modal.py --input-path ./statement.pdf --query "What is my total spend?"'
        )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File does not exist: {input_path}")

    filename = os.path.basename(input_path)
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    print(f"\n{'=' * 60}")
    print(f"  FINANCIAL DOCUMENT INSIGHTS ENGINE")
    print(f"{'=' * 60}")
    print(f"  File  : {filename}")
    print(f"  Query : {query}")
    print(f"{'=' * 60}\n")

    engine = FinancialInsightsEngine()
    result = engine.run_pipeline.remote(
        input_bytes = input_bytes,
        filename    = filename,
        query       = query,
    )

    print(f"\n{'=' * 60}")
    print(f"  FINAL ANSWER")
    print(f"{'=' * 60}")
    print(f"  Summary : {result['summary']}")
    print(f"{'=' * 60}")
    print(f"\n{result['answer']}")
    print(f"\n{'=' * 60}\n")