"""
Code Generator Node
--------------------
Takes the user query + extraction schema + extracted data and generates
Python code to compute the answer.

The generated code:
- Receives `data` as a list of dicts (transactions)
- Receives `statement` as a dict (document-level fields)
- Must assign the final answer to a variable called `result`
"""

import sys
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from schema_planner import schema_planner_node, ExtractionSchema
from extractor import extractor_node
from router import router_node

client = OpenAI()
MODEL = "gpt-5-mini-2025-08-07"  # swap to gpt-5-mini when available


# ── Pydantic models ────────────────────────────────────────────────────────────

class GeneratedCode(BaseModel):
    code      : str = Field(description="Valid Python code string that computes the answer. Must assign final answer to a variable called `result`.")
    reasoning : str = Field(description="Brief explanation of what the code does and why")


# ── Node function ──────────────────────────────────────────────────────────────

def code_generator_node(
    query       : str,
    schema      : ExtractionSchema,
    statement   : dict,
    transactions: list[dict],
    error       : str | None = None,
) -> GeneratedCode:
    """
    Generates Python code to compute the answer to the query
    given the extracted statement and transactions data.
    If error is provided, it means previous code failed and needs fixing.
    """

    error_context = (
        f"\n\nThe previously generated code failed with this error:\n{error}\n"
        "Please fix the code and try again."
        if error else ""
    )

    response = client.responses.parse(
        model  = MODEL,
        input  = [
            {
                "role"    : "system",
                "content" : (
                    "You are a Python code generator for financial data analysis. "
                    "You will be given a user query, extracted financial data, and a schema. "
                    "Your job is to write clean Python code that computes the answer to the query.\n\n"
                    "Rules:\n"
                    "- You have access to two variables: `data` (list of dicts) and `statement` (dict).\n"
                    "- Always assign the final answer to a variable called `result`.\n"
                    "- You may use pandas, but prefer plain Python where possible.\n"
                    "- Do not import anything other than pandas or standard library modules.\n"
                    "- Do not hardcode any values — use the data provided.\n"
                    "- Be strict about boundary conditions — match the query exactly e.g. >= vs >, <= vs <.\n"
                    "- Handle null/None values gracefully — always use .get() to access dict keys and skip records where required fields are None.\n"
                    "- `result` must be a string, number, or list — never a DataFrame or complex object.\n"
                    "- Do not print anything — just assign to `result`."
                )
            },
            {
                "role"    : "user",
                "content" : (
                    f"Query: {query}\n\n"
                    f"Computation hint: {schema.computation_hint}\n\n"
                    f"Statement (document-level fields):\n{json.dumps(statement, indent=2)}\n\n"
                    f"Transactions (list of dicts):\n{json.dumps(transactions, indent=2)}"
                    f"{error_context}"
                )
            }
        ],
        text_format = GeneratedCode,
    )

    return response.output_parsed # type: ignore


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python code_generator.py <path_to_markdown> "<query>"')
        sys.exit(1)

    md_path  = sys.argv[1]
    query    = sys.argv[2]
    markdown = open(md_path, encoding="utf-8").read()

    print(f"Query   : {query}")
    print(f"MD file : {md_path}\n")

    print("[1/4] Running schema planner...")
    schema = schema_planner_node(query, markdown)
    print(f"      → {len(schema.fields)} fields planned\n")

    print("[2/4] Running extractor...")
    statement, transactions = extractor_node(query, schema, markdown)
    print(f"      → {len(transactions)} transaction(s) extracted\n")

    print("[3/4] Running router...")
    decision = router_node(query, statement, transactions)
    print(f"      → needs_computation: {decision.needs_computation}")
    print(f"      → reason: {decision.reason}\n")

    if not decision.needs_computation:
        print("  [SKIP] Code generation not needed for this query.")
        sys.exit(0)

    print("[4/4] Running code generator...")
    generated = code_generator_node(query, schema, statement, transactions)

    print("\n── Reasoning ─────────────────────────────────────────────────────")
    print(generated.reasoning)

    print("\n── Generated Code ────────────────────────────────────────────────")
    print(generated.code)
    print("\n── Statement ─────────────────────────────────────────────────────")
    print(json.dumps(statement, indent=2))

    print(f"\n── All Transactions ({len(transactions)}) ────────────────────────────────────")
    for i, tx in enumerate(transactions, 1):
        print(f"  [{i:02d}] {json.dumps(tx)}")

    print(f"\n── Schema Fields ({len(schema.fields)}) ──────────────────────────────────────")
    for f in schema.fields:
        print(f"  {f.name} ({f.type}) — {f.description} [required={f.required}]")
    print(f"\n  Filter hint      : {schema.filter_hint}")
    print(f"  Computation hint : {schema.computation_hint}")
    print(f"  Document context : {schema.document_context}")