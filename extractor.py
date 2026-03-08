"""
Extractor Node
---------------
Takes the query + schema from Schema Planner + entire markdown and extracts
structured data from the financial document.

Returns:
    ExtractedDocument with:
    - statement    : dict       — document-level fields parsed from JSON string
    - transactions : list[dict] — transaction records parsed from JSON string
"""

import sys
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from schema_planner import schema_planner_node, ExtractionSchema

client = OpenAI()
MODEL = "gpt-5-mini-2025-08-07"  # swap to gpt-5-mini when available


# ── Pydantic models ────────────────────────────────────────────────────────────

class ExtractedDocument(BaseModel):
    statement: str = Field(
        description="Valid JSON object string of document-level summary fields e.g. dates, totals, limits, account holder. Use {} if none found.")
    transactions: str = Field(
        description="Valid JSON array string of transaction records. Each record contains only the fields from the schema. Use [] if none found.")


# ── Node function ──────────────────────────────────────────────────────────────


def _safe_parse_json(raw: str, expected: str = "object"):
    """
    Robustly parse a JSON string returned by the LLM.
    - Strips markdown fences
    - Extracts first valid JSON object {} or array [] using regex
    """
    import re

    # Strip markdown fences
    cleaned = raw.strip().strip("```json").strip("```").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract first { ... } or [ ... ] block
    if expected == "object":
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    else:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Return safe default
    return {} if expected == "object" else []


def extractor_node(query: str, schema: ExtractionSchema, markdown: str) -> tuple[dict, list[dict]]:
    """
    Given the user query, extraction schema and full markdown,
    extracts all relevant data. Returns (statement_dict, transactions_list).
    """
    fields_description = "\n".join([
        f"  - {f.name} ({f.type}): {f.description} [required={f.required}]"
        for f in schema.fields
    ])

    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a financial document data extractor. "
                    "You will receive a markdown version of a financial document and a schema to extract.\n\n"

                    "The document may be a credit card statement, bank statement, mutual fund statement, "
                    "loan statement, invoice, or any other financial document.\n\n"

                    "Extract data strictly according to the provided schema.\n\n"

                    "Rules:\n"
                    "- Extract values only from the markdown. Do not infer, assume, calculate or hallucinate.\n"
                    "- Do not assume any specific document type, currency, or table structure.\n"
                    "- Put document-level summary fields (totals, dates, limits, etc.) in 'statement' as a JSON object string.\n"
                    "- Put row-level records in 'transactions' as a JSON array string.\n"
                    "- If a field is missing, return null.\n"
                    "- Amounts must be floats. Remove currency symbols and commas (e.g., 5,800.00 → 5800.0).\n"
                    "- Dates must be returned exactly as they appear in the document.\n"
                    "- 'statement' and 'transactions' must be valid JSON strings with no extra text, markdown, or trailing commas.\n\n"

                    f"Filter hint: {schema.filter_hint}\n"
                    f"Original query: {query}\n\n"

                    "Important: Do not infer, estimate, compute, or derive any value on your own. "
                    "Only return values that are explicitly present in the markdown."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Schema fields to extract:\n{fields_description}\n\n"
                    f"Document context: {schema.document_context}\n\n"
                    f"Markdown:\n{markdown}"
                )
            }
        ],
        text_format=ExtractedDocument,
    )

    raw = response.output_parsed

    statement = _safe_parse_json(
        raw.statement, expected="object")  # type: ignore
    transactions = _safe_parse_json(
        raw.transactions, expected="array")  # type: ignore

    return statement, transactions  # type: ignore


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python extractor.py <path_to_markdown> "<query>"')
        sys.exit(1)

    md_path = sys.argv[1]
    query = sys.argv[2]
    markdown = open(md_path, encoding="utf-8").read()

    print(f"Query   : {query}")
    print(f"MD file : {md_path}\n")

    print("[1/2] Running schema planner...")
    schema = schema_planner_node(query, markdown)
    print(f"      → {len(schema.fields)} fields planned\n")

    print("[2/2] Running extractor...")
    statement, transactions = extractor_node(query, schema, markdown)

    print("\n── Statement ─────────────────────────────────────────────────────")
    print(json.dumps(statement, indent=2))

    print(
        f"\n── Transactions ({len(transactions)}) ──────────────────────────────────────")
    for i, tx in enumerate(transactions, 1):
        print(f"  [{i:02d}] {tx}")
