"""
Schema Planner Node
--------------------
Takes the user query + entire markdown and decides:
- What fields to extract from the document
- Any filter hints (e.g. only debit transactions)
- A computation hint for the code generator
"""

from pydantic import BaseModel, Field
from openai import OpenAI
import os

client = OpenAI()

MODEL = "gpt-5-mini-2025-08-07"


# ── Pydantic schema ────────────────────────────────────────────────────────────

class FieldDefinition(BaseModel):
    name: str = Field(
        description="Field name in snake_case e.g. transaction_date")
    type: str = Field(
        description="Python type as string: str, float, int, bool")
    description: str = Field(description="What this field represents")
    required: bool = Field(description="Whether this field is always expected")


class ExtractionSchema(BaseModel):
    fields: list[FieldDefinition] = Field(
        description="Fields to extract per transaction or record")
    filter_hint: str = Field(
        description="What to include/exclude e.g. 'only debit transactions'")
    computation_hint: str = Field(
        description="How to compute the final answer e.g. 'sum amount where category is shopping'")
    document_context: str = Field(
        description="Brief description of document type and period inferred from markdown")


# ── Node function ──────────────────────────────────────────────────────────────

def schema_planner_node(query: str, markdown: str) -> ExtractionSchema:
    """
    Given a user query and the full markdown of a financial document,
    returns a structured schema describing what to extract.
    """
    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a financial document analyst. "
                    "You will be given a user query and a markdown representation of a financial document. "
                    "The document may be a credit card statement, bank statement, mutual fund statement, "
                    "loan statement, invoice, or any other financial document. "
                    "Your task is to determine which fields must be extracted from the document "
                    "in order to answer the user's query. "
                    "Be precise — only include fields that are necessary to answer the query. "
                    "Always include supporting context fields like date, description, and merchant name (if they exist) "
                    "alongside computation fields so the result is meaningful. "
                    "Do not assume a specific document type, currency, schema, or structure. "
                    "Derive field names, data types, and descriptions directly from the document content. "
                    "All field names must be written in snake_case. "
                    "If the document contains transaction amounts with CR, C, or credit suffixes, "
                    "define the amount field as a positive float and include a separate boolean or string field "
                    "to indicate whether the transaction is a credit (payment/refund) or debit (purchase/charge). "
                    "Never define amount fields as negative — credits are not negative amounts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Document:\n{markdown}"
                )
            }
        ],
        text_format=ExtractionSchema,
    )

    return response.output_parsed  # type: ignore


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python schema_planner.py <path_to_markdown> \"<query>\"")
        sys.exit(1)

    md_path = sys.argv[1]
    query = sys.argv[2]

    sample_markdown = open(md_path, encoding="utf-8").read()

    print(f"Query    : {query}")
    print(f"MD file  : {md_path}\n")

    schema = schema_planner_node(query, sample_markdown)

    print("Fields:")
    for f in schema.fields:
        print(
            f"  {f.name} ({f.type}) — {f.description} [required={f.required}]")
    print(f"\nFilter hint      : {schema.filter_hint}")
    print(f"Computation hint : {schema.computation_hint}")
    print(f"Document context : {schema.document_context}")
