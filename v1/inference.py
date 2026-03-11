"""
inference.py
------------
Client script to call the FinancialInsightsEngine FastAPI endpoint.

Usage:
    python inference.py --url <modal_endpoint_url> --input-path ./statement.pdf --query "What is my total spend?"
"""

import os
import sys
import base64
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx"}
DIVIDER = "─" * 60


def encode_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_endpoint(url: str, file_path: str, query: str) -> dict:
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    print(f"\n{'=' * 60}")
    print(f"  FINANCIAL DOCUMENT INSIGHTS ENGINE")
    print(f"{'=' * 60}")
    print(f"  File     : {filename}")
    print(f"  Query    : {query}")
    print(f"  Endpoint : {url}")
    print(f"{'=' * 60}\n")

    print(f"[INFO] Encoding file...")
    file_b64 = encode_file(file_path)
    print(f"[INFO] File size : {len(file_b64) / 1024:.1f} KB (base64)")

    payload = {
        "file_b64": file_b64,
        "filename": filename,
        "query": query,
    }

    print(f"[INFO] Sending request to endpoint...")
    response = requests.post(url, json=payload, timeout=600)

    if response.status_code != 200:
        print(f"[ERROR] Request failed with status {response.status_code}")
        print(f"[ERROR] {response.text}")
        sys.exit(1)

    return response.json()


def print_result(result: dict):
    # ── Step 1 : Schema Planner ───────────────────────────────────
    schema = result.get("schema", {})
    fields = schema.get("fields", [])
    print(f"\n{'=' * 60}")
    print(f"  [1/6] SCHEMA PLANNER")
    print(f"{'=' * 60}")
    print(f"  Fields planned : {len(fields)}")
    for f in fields:
        print(f"    • {f['name']} ({f['type']}) — {f['description']}")
    print(f"  Filter hint      : {schema.get('filter_hint', '—')}")
    print(f"  Computation hint : {schema.get('computation_hint', '—')}")
    print(f"  Document context : {schema.get('document_context', '—')}")

    # ── Step 2 : Extractor ────────────────────────────────────────
    statement = result.get("statement", {})
    transactions = result.get("transactions", [])
    print(f"\n{'=' * 60}")
    print(f"  [2/6] EXTRACTOR")
    print(f"{'=' * 60}")
    print(f"  Statement fields : {list(statement.keys())}")
    for k, v in statement.items():
        print(f"    • {k} : {v}")
    print(f"  Transactions extracted : {len(transactions)}")
    for i, tx in enumerate(transactions, 1):
        print(f"    [{i:02d}] {tx}")

    # ── Step 3 : Router ───────────────────────────────────────────
    router = result.get("router", {})
    print(f"\n{'=' * 60}")
    print(f"  [3/6] ROUTER")
    print(f"{'=' * 60}")
    print(f"  Needs computation : {router.get('needs_computation', '—')}")
    print(f"  Reason            : {router.get('reason', '—')}")

    # ── Step 4 : Code Generator ───────────────────────────────────
    cg = result.get("code_generator", {})
    code = cg.get("code")
    print(f"\n{'=' * 60}")
    print(f"  [4/6] CODE GENERATOR")
    print(f"{'=' * 60}")
    if code:
        print(f"  Reasoning : {cg.get('reasoning', '—')}")
        print(f"  Generated code:")
        for line in code.splitlines():
            print(f"    {line}")
    else:
        print(f"  SKIPPED — no computation needed")

    # ── Step 5 : Executor ─────────────────────────────────────────
    executor = result.get("executor", {})
    success = executor.get("success")
    print(f"\n{'=' * 60}")
    print(f"  [5/6] EXECUTOR")
    print(f"{'=' * 60}")
    if success is None:
        print(f"  SKIPPED — no computation needed")
    elif success:
        print(f"  ✓ Execution successful")
        print(f"  Result : {executor.get('result', '—')}")
    else:
        print(f"  ✗ Execution failed")
        print(f"  Error  : {executor.get('error', '—')}")

    # ── Step 6 : Narrator (Final Answer) ──────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  [6/6] FINAL ANSWER")
    print(f"{'=' * 60}")
    print(f"  Summary : {result.get('summary', 'N/A')}")
    print(f"{'=' * 60}")
    print(f"\n{result.get('answer', 'No answer returned.')}")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Financial Document Insights Engine — inference client")
    parser.add_argument("--url",        required=False, default=None,
                        help="Modal FastAPI endpoint URL (overrides .env)")
    parser.add_argument("--input-path", required=True,
                        help="Path to input file (PDF, CSV, or XLSX)")
    parser.add_argument("--query",      required=True,
                        help='Query e.g. "What is my total spend?"')
    args = parser.parse_args()

    url = args.url or os.getenv("INFERENCE_ENDPOINT")
    if not url:
        print(
            "[ERROR] No endpoint URL provided. Set INFERENCE_ENDPOINT in .env or pass --url")
        sys.exit(1)

    if not os.path.exists(args.input_path):
        print(f"[ERROR] File not found: {args.input_path}")
        sys.exit(1)

    result = call_endpoint(
        url=url,
        file_path=args.input_path,
        query=args.query,
    )

    print_result(result)
