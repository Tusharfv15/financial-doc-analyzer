"""
Financial Document Insights Engine — Full Pipeline
----------------------------------------------------
Two modes:

  --rag    : PDF -> Markdown -> Chunk -> Embed -> Retrieve -> Insights
  --direct : PDF -> Markdown -> Insights (full document, no chunking)

Usage:
    python financial_assistant.py <path_to_pdf> "<query>" --rag
    python financial_assistant.py <path_to_pdf> "<query>" --direct
    python financial_assistant.py <path_to_pdf> "<query>"   # prompts to choose

Examples:
    python financial_assistant.py docs/icici.pdf "What is my total spend?" --rag
    python financial_assistant.py docs/icici.pdf "Show me all EMI transactions" --direct

Environment variables required:
    OPENAI_API_KEY
    PINECONE_API_KEY        (RAG mode only)
    PINECONE_INDEX_NAME     (RAG mode only, default: jupiter-chunks)
"""

import sys
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from pydantic import BaseModel, Field
from openai import OpenAI

from schema_planner import schema_planner_node
from extractor      import extractor_node
from router         import router_node
from code_generator import code_generator_node
from code_executor  import executor_node, MAX_RETRIES
from retriever      import retrieve_chunks
from embed_v2        import embed_document
from chunker_v2       import chunk_file

client  = OpenAI()
MODEL   = "gpt-5-mini-2025-08-07"
DIVIDER = "─" * 60


# ══════════════════════════════════════════════════════════════════
#  PDF -> Markdown
# ══════════════════════════════════════════════════════════════════

def convert_pdf_to_md(pdf_path: str) -> str:
    from docling.document_converter import DocumentConverter

    pdf_path  = Path(pdf_path) # type: ignore
    if not pdf_path.exists(): # type: ignore
        raise FileNotFoundError(f"File does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf": # type: ignore
        raise ValueError("Provided file is not a PDF")

    output_dir = Path("docling_md")
    output_dir.mkdir(exist_ok=True)
    output_md  = output_dir / f"{pdf_path.stem}.md" # type: ignore

    if output_md.exists():
        print(f"  [SKIP] Markdown already exists: {output_md}")
        return str(output_md)

    print(f"  Input PDF  : {pdf_path}")
    print(f"  Converting via Docling...")
    start     = time.time()
    converter = DocumentConverter()
    result    = converter.convert(str(pdf_path))
    markdown  = result.document.export_to_markdown()

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"  Saved : {output_md}  ({round(time.time() - start, 2)}s)")
    return str(output_md)


# ══════════════════════════════════════════════════════════════════
#  Narrator Node
# ══════════════════════════════════════════════════════════════════

class NarratorResponse(BaseModel):
    answer  : str = Field(description="Clear, concise natural language answer to the user's query")
    summary : str = Field(description="One-line summary of the key insight e.g. 'Total spend: INR 45,000'")


def narrator_node(
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

    response = client.responses.parse(
        model  = MODEL,
        input  = [
            {
                "role"    : "system",
                "content" : (
                    "You are a helpful financial document assistant. "
                    "Your job is to narrate the answer clearly and naturally.\n\n"
                    "Rules:\n"
                    "- Be concise and direct — answer the query first, then provide context.\n"
                    "- Always mention the currency if present in the data.\n"
                    "- If the result is a list of transactions, present them in a readable format.\n"
                    "- If the result is a number, round to 2 decimal places.\n"
                    "- Highlight any notable observations.\n"
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


# ══════════════════════════════════════════════════════════════════
#  Shared Insights Pipeline
# ══════════════════════════════════════════════════════════════════

def run_insights_pipeline(query: str, markdown: str, mode_label: str):
    print(f"\n{'=' * 60}")
    print(f"  INSIGHTS PIPELINE  [{mode_label}]")
    print(f"{'=' * 60}")
    print(f"  Query    : {query}")
    print(f"  Markdown : {len(markdown)} chars")
    print(f"{'=' * 60}\n")

    print(f"[1/6] Schema Planner")
    print(f"      {DIVIDER}")
    schema = schema_planner_node(query, markdown)
    print(f"  Fields planned : {len(schema.fields)}")
    for f in schema.fields:
        print(f"    • {f.name} ({f.type}) — {f.description}")
    print(f"  Filter hint    : {schema.filter_hint}")
    print(f"  Compute hint   : {schema.computation_hint}")
    print(f"  Doc context    : {schema.document_context}")
    print()

    print(f"[2/6] Extractor")
    print(f"      {DIVIDER}")
    statement, transactions = extractor_node(query, schema, markdown)
    print(f"  Transactions extracted : {len(transactions)}")
    print(f"  Statement values:")
    for k, v in statement.items():
        print(f"    • {k} : {v}")
    print(f"  Transactions:")
    for i, tx in enumerate(transactions, 1):
        print(f"    [{i:02d}] {json.dumps(tx)}")
    print()

    print(f"[3/6] Router")
    print(f"      {DIVIDER}")
    decision = router_node(query, statement, transactions)
    print(f"  Needs computation : {decision.needs_computation}")
    print(f"  Reason            : {decision.reason}")
    print()

    exec_result = None

    if decision.needs_computation:
        print(f"[4/6] Code Generator")
        print(f"      {DIVIDER}")
        generated = code_generator_node(query, schema, statement, transactions)
        print(f"  Reasoning : {generated.reasoning}")
        print(f"  Generated code:")
        for line in generated.code.splitlines():
            print(f"    {line}")
        print()

        print(f"[5/6] Executor")
        print(f"      {DIVIDER}")
        exec_result = executor_node(query, schema, statement, transactions, generated)
        if exec_result.success:
            print(f"  Execution successful")
            print(f"  Raw result : {exec_result.result}")
        else:
            print(f"  Execution failed after {MAX_RETRIES} attempts")
            print(f"  Error : {exec_result.error}")
        print()
    else:
        print(f"[4/6] Code Generator  — SKIPPED")
        print(f"[5/6] Executor        — SKIPPED")
        print()

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


# ══════════════════════════════════════════════════════════════════
#  RAG mode — index + retrieve + insights
# ══════════════════════════════════════════════════════════════════

def run_rag(pdf_path: str, query: str):
    pdf_path = Path(pdf_path) # type: ignore
    pdf_name = pdf_path.stem # type: ignore

    print(f"\n{'=' * 60}")
    print(f"  RAG MODE")
    print(f"{'=' * 60}\n")

    # Step 1 — PDF -> Markdown
    print(f"[1/3] PDF -> Markdown")
    print(f"      {DIVIDER}")
    md_path = convert_pdf_to_md(str(pdf_path))
    print()

    # Step 2 — Chunk + Embed + Index
    print(f"[2/3] Chunking + Embedding + Indexing")
    print(f"      {DIVIDER}")
    chunks_dir = Path("doc_chunks") / pdf_name
    if chunks_dir.exists() and any(chunks_dir.glob("chunk_*.md")):
        print(f"  [SKIP] Chunks already exist: {chunks_dir}")
    else:
        chunk_file(md_path)
    embed_document(pdf_name)
    print()

    # Step 3 — Retrieve
    print(f"[3/3] Retrieving relevant chunks")
    print(f"      {DIVIDER}")
    markdown, chunks_info = retrieve_chunks(query, pdf_name)

    if not markdown:
        print(f"  [ERROR] No relevant chunks found for this query.")
        sys.exit(1)

    print(f"  Retrieved {len(chunks_info)} chunk(s):")
    for c in chunks_info:
        print(f"    chunk_{c['chunk_id']} | score: {c['score']} | {c['summary']}")
    print()

    # Save merged markdown
    merged_dir = Path("merged_markdown")
    merged_dir.mkdir(exist_ok=True)
    merged_path = merged_dir / f"{pdf_name}-merged.md"
    merged_path.write_text(markdown, encoding="utf-8")
    print(f"  Merged markdown saved : {merged_path}")
    print()

    run_insights_pipeline(query, markdown, mode_label="RAG mode")


# ══════════════════════════════════════════════════════════════════
#  Direct mode — full document -> insights
# ══════════════════════════════════════════════════════════════════

def run_direct(pdf_path: str, query: str):
    print(f"\n{'=' * 60}")
    print(f"  DIRECT MODE")
    print(f"{'=' * 60}\n")

    print(f"[1/1] PDF -> Markdown")
    print(f"      {DIVIDER}")
    md_path  = convert_pdf_to_md(pdf_path)
    markdown = open(md_path, encoding="utf-8").read()
    print()

    run_insights_pipeline(query, markdown, mode_label="Direct mode")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def print_usage():
    print("""
Usage:
  python financial_assistant.py <path_to_pdf> "<query>" --rag
  python financial_assistant.py <path_to_pdf> "<query>" --direct
  python financial_assistant.py <path_to_pdf> "<query>"    <- prompts to choose

Modes:
  --rag    Index the document into Pinecone, retrieve relevant chunks, run insights
  --direct Send full document directly to insights pipeline (no indexing)

Examples:
  python financial_assistant.py docs/icici.pdf "What is my total spend?" --rag
  python financial_assistant.py docs/icici.pdf "Show me all EMI transactions" --direct
""")


def prompt_mode() -> str:
    print("\n  Choose mode:")
    print("  [1] RAG    — chunk + index into Pinecone, then retrieve relevant chunks")
    print("  [2] Direct — send full document to pipeline (no indexing needed)")
    while True:
        choice = input("\n  Enter 1 or 2: ").strip()
        if choice == "1":
            return "rag"
        elif choice == "2":
            return "direct"
        else:
            print("  Invalid. Please enter 1 or 2.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    pdf_path = sys.argv[1]
    query    = sys.argv[2]
    flags    = sys.argv[3:]

    if "--rag" in flags:
        mode = "rag"
    else:
        mode = "direct"

    if mode == "rag":
        run_rag(pdf_path, query)
    else:
        run_direct(pdf_path, query)