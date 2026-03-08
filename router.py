"""
Router Node
------------
Decides whether the query needs code generation or can be answered
directly from the extracted data.

Returns:
    RouterDecision with:
    - needs_computation : bool   — True if code generation is needed
    - reason            : str    — brief explanation of the decision
"""

import sys
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from schema_planner import schema_planner_node, ExtractionSchema
from extractor import extractor_node

client = OpenAI()
MODEL = "gpt-5-mini-2025-08-07"  # swap to gpt-5-mini when available


# ── Pydantic models ────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    needs_computation: bool = Field(description=(
        "True if the query requires computation e.g. aggregation, sum, average, percentage, "
        "groupby, ranking, trend comparison. "
        "False if the query can be answered by simply returning the extracted data as-is."
    ))
    reason: str = Field(
        description="Brief one-line explanation of the routing decision")


# ── Node function ──────────────────────────────────────────────────────────────

def router_node(
    query: str,
    statement: dict,
    transactions: list[dict],
) -> RouterDecision:
    """
    Given the query and extracted data, decides whether code generation
    is needed or if the data can be returned directly to the narrator.
    """
    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a query router for a financial document analysis pipeline. "
                    "You will be given a user query and extracted financial data. "
                    "Your job is to decide whether the query requires computation or not.\n\n"
                    "Needs computation (needs_computation=True):\n"
                    "- Aggregations: sum, total, average, count\n"
                    "- Rankings: top N merchants, highest spend category\n"
                    "- Ratios or percentages: credit utilization, spend ratio\n"
                    "- Comparisons: this month vs last month\n"
                    "- Grouping: category-wise spend, merchant-wise breakdown\n\n"
                    "Does NOT need computation (needs_computation=False):\n"
                    "- Listing or filtering: show me all transactions above X\n"
                    "- Single value lookup: what is my statement date, what is my credit limit\n"
                    "- Simple retrieval: list all EMI transactions, show recurring payments\n"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Statement fields available:\n{json.dumps(statement, indent=2)}\n\n"
                    f"Sample of extracted transactions ({min(3, len(transactions))} of {len(transactions)}):\n"
                    f"{json.dumps(transactions[:3], indent=2)}"
                )
            }
        ],
        text_format=RouterDecision,
    )

    return response.output_parsed  # type: ignore


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python router.py <path_to_markdown> "<query>"')
        sys.exit(1)

    md_path = sys.argv[1]
    query = sys.argv[2]
    markdown = open(md_path, encoding="utf-8").read()

    print(f"Query   : {query}")
    print(f"MD file : {md_path}\n")

    print("[1/3] Running schema planner...")
    schema = schema_planner_node(query, markdown)
    print(f"      → {len(schema.fields)} fields planned\n")

    print("[2/3] Running extractor...")
    statement, transactions = extractor_node(query, schema, markdown)
    print(f"      → {len(transactions)} transaction(s) extracted\n")

    print("[3/3] Running router...")
    decision = router_node(query, statement, transactions)

    print(f"\n── Router Decision ───────────────────────────────────────────────")
    print(f"  Needs computation : {decision.needs_computation}")
    print(f"  Reason            : {decision.reason}")
    print(
        f"\n  → Next node : {'Code Generator' if decision.needs_computation else 'Narrator'}")
