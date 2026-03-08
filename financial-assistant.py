"""
Narrator Node
--------------
Takes the query + extracted data + execution result and narrates
the final answer in a natural, human-readable way.

Works in two modes:
- Computation mode : narrates the executor result (number, list, etc.)
- Direct mode      : narrates the extracted transactions directly (no code ran)
"""

import sys
import json
import time
import os
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from docling.document_converter import DocumentConverter
from schema_planner import schema_planner_node, ExtractionSchema
from extractor import extractor_node
from router import router_node
from code_generator import code_generator_node
from code_executor import executor_node, ExecutorResult, MAX_RETRIES

client  = OpenAI()
MODEL   = "gpt-5-mini-2025-08-07"
DIVIDER = "─" * 60

# Disable symlinks for HuggingFace Hub
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


# ── PDF to Markdown Conversion ─────────────────────────────────────────────────

def convert_pdf_to_md(pdf_path: str) -> str:
    """
    Converts a PDF to markdown using Docling.
    Returns the path to the generated markdown file.
    """
    start = time.time()
    pdf_path = Path(pdf_path) # type: ignore

    # ---- Input validation ----
    if not pdf_path.exists(): # type: ignore
        raise FileNotFoundError(f"File does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf": # type: ignore
        raise ValueError("Provided file is not a PDF")
    # --------------------------

    output_dir = Path("docling_md")
    output_dir.mkdir(exist_ok=True)

    output_md = output_dir / f"{pdf_path.stem}.md" # type: ignore

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
    print(f"[INFO] Total time: {round(end - start, 2)} seconds\n")

    return str(output_md)


# ── Pydantic models ────────────────────────────────────────────────────────────

class NarratorResponse(BaseModel):
    answer  : str = Field(description="Clear, concise natural language answer to the user's query")
    summary : str = Field(description="One-line summary of the key insight e.g. 'Total spend: PHP 45,000'")


# ── Node function ──────────────────────────────────────────────────────────────

def narrator_node(
    query        : str,
    statement    : dict,
    transactions : list[dict],
    result       : str | None = None,
) -> NarratorResponse:
    """
    Narrates the final answer to the user.
    - If result is provided, narrates the computed result.
    - If result is None, narrates the extracted transactions directly.
    """

    data_context = (
        f"Computed result:\n{result}"
        if result else
        f"Extracted records ({len(transactions)}):\n{json.dumps(transactions, indent=2)}"
    )

    response = client.responses.parse(
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

    return response.output_parsed # type: ignore


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python narrator.py <path_to_pdf_or_markdown> "<query>"')
        print('  - If PDF is provided, it will be converted to markdown first')
        print('  - If markdown is provided, it will be used directly')
        sys.exit(1)

    input_path = sys.argv[1]
    query      = sys.argv[2]

    # Check if input is PDF or Markdown
    input_path_obj = Path(input_path)
    
    if not input_path_obj.exists():
        print(f"[ERROR] File does not exist: {input_path}")
        sys.exit(1)

    if input_path_obj.suffix.lower() == ".pdf":
        # Convert PDF to markdown
        print(f"\n{'=' * 60}")
        print(f"  PDF TO MARKDOWN CONVERSION")
        print(f"{'=' * 60}\n")
        
        md_path = convert_pdf_to_md(input_path)
        markdown = open(md_path, encoding="utf-8").read()
        
    elif input_path_obj.suffix.lower() == ".md":
        # Use markdown directly
        md_path = input_path
        markdown = open(md_path, encoding="utf-8").read()
        
    else:
        print(f"[ERROR] Unsupported file type. Please provide a PDF or Markdown file.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  FINANCIAL DOCUMENT INSIGHTS ENGINE")
    print(f"{'=' * 60}")
    print(f"  Query   : {query}")
    print(f"  MD file : {md_path}")
    print(f"{'=' * 60}\n")

    # ── Step 1 ─────────────────────────────────────────────────────
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

    # ── Step 2 ─────────────────────────────────────────────────────
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

    # ── Step 3 ─────────────────────────────────────────────────────
    print(f"[3/6] Router")
    print(f"      {DIVIDER}")
    decision = router_node(query, statement, transactions)
    print(f"      Needs computation : {decision.needs_computation}")
    print(f"      Reason            : {decision.reason}")
    print()

    exec_result = None

    if decision.needs_computation:
        # ── Step 4 ───────────────────────────────────────────────
        print(f"[4/6] Code Generator")
        print(f"      {DIVIDER}")
        generated = code_generator_node(query, schema, statement, transactions)
        print(f"      Reasoning : {generated.reasoning}")
        print(f"      Generated code:")
        for line in generated.code.splitlines():
            print(f"        {line}")
        print()

        # ── Step 5 ───────────────────────────────────────────────
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

    # ── Step 6 ─────────────────────────────────────────────────────
    print(f"[6/6] Narrator")
    print(f"      {DIVIDER}")
    narrator = narrator_node(
        query        = query,
        statement    = statement,
        transactions = transactions,
        result       = exec_result.result if exec_result and exec_result.success else None,
    )

    print(f"\n{'=' * 60}")
    print(f"  FINAL ANSWER")
    print(f"{'=' * 60}")
    print(f"  Summary : {narrator.summary}")
    print(f"{'=' * 60}")
    print(f"\n{narrator.answer}")
    print(f"\n{'=' * 60}\n")