"""
Executor Node
--------------
Executes the generated Python code in a restricted sandbox.
Has access to `data` (transactions) and `statement` (document-level fields).
Retries up to MAX_RETRIES times by passing the error back to the code generator.

Returns:
    ExecutorResult with:
    - success : bool   — whether execution succeeded
    - result  : str    — the value assigned to `result` variable
    - error   : str    — error message if execution failed
"""

import sys
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from schema_planner import schema_planner_node, ExtractionSchema
from extractor import extractor_node
from router import router_node
from code_generator import code_generator_node, GeneratedCode

client      = OpenAI()
MODEL       = "gpt-5-mini-2025-08-07"
MAX_RETRIES = 2

DIVIDER = "─" * 60


# ── Pydantic models ────────────────────────────────────────────────────────────

class ExecutorResult(BaseModel):
    success : bool
    result  : str = Field(default="", description="Final result as a string")
    error   : str = Field(default="", description="Error message if execution failed")


# ── Node function ──────────────────────────────────────────────────────────────

def executor_node(
    query        : str,
    schema       : ExtractionSchema,
    statement    : dict,
    transactions : list[dict],
    generated    : GeneratedCode,
) -> ExecutorResult:
    """
    Executes the generated code in a sandbox.
    Retries up to MAX_RETRIES times if execution fails.
    """
    error = None

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"  [Retry {attempt}/{MAX_RETRIES}] Regenerating code after error...")
            generated = code_generator_node(query, schema, statement, transactions, error=error)
            print(f"  [Retry {attempt}/{MAX_RETRIES}] New code generated")

        try:
            sandbox = {
                "data"      : transactions,
                "statement" : statement,
            }

            exec(generated.code, sandbox)

            if "result" not in sandbox:
                raise ValueError("Code did not assign to `result` variable.")

            raw = sandbox["result"]

            if isinstance(raw, (dict, list)):
                result_str = json.dumps(raw, indent=2)
            else:
                result_str = str(raw)

            return ExecutorResult(success=True, result=result_str)

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n\nCode that failed:\n{generated.code}"
            print(f"  [ERROR] Attempt {attempt} failed: {type(e).__name__}: {e}")

    return ExecutorResult(
        success = False,
        result  = "",
        error   = f"All {MAX_RETRIES} attempts failed. Last error:\n{error}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python executor.py <path_to_markdown> "<query>"')
        sys.exit(1)

    md_path  = sys.argv[1]
    query    = sys.argv[2]
    markdown = open(md_path, encoding="utf-8").read()

    print(f"\n{'=' * 60}")
    print(f"  FINANCIAL DOCUMENT INSIGHTS ENGINE")
    print(f"{'=' * 60}")
    print(f"  Query   : {query}")
    print(f"  MD file : {md_path}")
    print(f"{'=' * 60}\n")

    # ── Step 1 ─────────────────────────────────────────────────────
    print(f"[1/5] Schema Planner")
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
    print(f"[2/5] Extractor")
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
    print(f"[3/5] Router")
    print(f"      {DIVIDER}")
    decision = router_node(query, statement, transactions)
    print(f"      Needs computation : {decision.needs_computation}")
    print(f"      Reason            : {decision.reason}")
    print()

    if not decision.needs_computation:
        print(f"  [SKIP] Code generation not needed — returning extracted data directly.\n")
        print(f"── Final Result {DIVIDER}")
        for i, tx in enumerate(transactions, 1):
            print(f"  [{i:02d}] {json.dumps(tx)}")
        sys.exit(0)

    # ── Step 4 ─────────────────────────────────────────────────────
    print(f"[4/5] Code Generator")
    print(f"      {DIVIDER}")
    generated = code_generator_node(query, schema, statement, transactions)
    print(f"      Reasoning : {generated.reasoning}")
    print(f"      Generated code:")
    for line in generated.code.splitlines():
        print(f"        {line}")
    print()

    # ── Step 5 ─────────────────────────────────────────────────────
    print(f"[5/5] Executor")
    print(f"      {DIVIDER}")
    exec_result = executor_node(query, schema, statement, transactions, generated)

    print(f"\n{'=' * 60}")
    print(f"  RESULT")
    print(f"{'=' * 60}")
    if exec_result.success:
        print(f"  ✓ Success")
        print(f"  {exec_result.result}")
    else:
        print(f"  ✗ Failed after {MAX_RETRIES} attempts")
        print(f"  {exec_result.error}")
    print(f"{'=' * 60}\n")