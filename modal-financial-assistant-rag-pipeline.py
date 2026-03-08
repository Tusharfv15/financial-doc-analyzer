"""
Modal script — Financial Document Insights Engine (RAG + Direct)
-----------------------------------------------------------------
Two modes:
  - RAG    : File -> Markdown -> Chunk -> Embed -> Retrieve -> Insights
  - Direct : File -> Markdown -> Insights (full document, no chunking)

Usage:
    # RAG mode
    modal run rag-based-financial-assistant-modal.py --input-path ./statement.pdf --query "What is my total spend?" --mode rag

    # Direct mode
    modal run rag-based-financial-assistant-modal.py --input-path ./statement.pdf --query "What is my total spend?" --mode direct
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

app = modal.App("financial-insights-engine-rag")

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
        "pinecone",
    ])
    .add_local_python_source("schema_planner")
    .add_local_python_source("extractor")
    .add_local_python_source("router")
    .add_local_python_source("code_generator")
    .add_local_python_source("code_executor")
    .add_local_python_source("retriever")
    .add_local_python_source("embed_v2")
    .add_local_python_source("chunker_v2")
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

data_volume = modal.Volume.from_name("financial-insights-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    file_b64 : str = Field(description="Base64-encoded file contents (PDF, CSV, or XLSX)")
    filename : str = Field(description="Original filename including extension e.g. 'statement.pdf'")
    query    : str = Field(description="Natural language query about the document")
    mode     : str = Field(default="direct", description="'rag' or 'direct'")


class NarratorResponse(BaseModel):
    answer  : str = Field(description="Clear, concise natural language answer to the user's query")
    summary : str = Field(description="One-line summary of the key insight e.g. 'Total spend: INR 45,000'")


# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=60 * 20,
    scaledown_window=120,
    volumes={"/data": data_volume},
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("pinecone-secret"),  # must have PINECONE_API_KEY + PINECONE_INDEX_NAME
    ],
)
class FinancialInsightsEngine:

    @modal.enter()
    def setup(self):
        from docling.document_converter import DocumentConverter

        os.environ["DOCLING_CACHE_DIR"] = "/data/docling-cache"
        os.makedirs("/data/docling-cache", exist_ok=True)
        os.makedirs("/data/results", exist_ok=True)
        os.makedirs("/data/chunks", exist_ok=True)
        os.makedirs("/data/merged", exist_ok=True)

        print("Initializing OpenAI client...")
        self.client = OpenAI()
        print("Initializing Docling converter...")
        self.converter = DocumentConverter()
        print("Ready.")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def convert_to_md(self, input_path: str) -> str:
        start      = time.time()
        input_path = Path(input_path) # type: ignore

        if not input_path.exists(): # type: ignore
            raise FileNotFoundError(f"File does not exist: {input_path}")

        ext = input_path.suffix.lower() # type: ignore
        if ext not in {".pdf", ".csv", ".xlsx"}:
            raise ValueError(f"Unsupported file type: {ext}.")

        output_dir = Path("/data/results")
        output_dir.mkdir(exist_ok=True)
        output_md  = output_dir / f"{input_path.stem}.md" # type: ignore

        # Skip conversion if markdown already exists
        if output_md.exists():
            print(f"  [SKIP] Markdown already exists: {output_md}")
            return str(output_md)

        print(f"  Input file : {input_path} ({ext})")
        print("  Starting conversion...")
        result   = self.converter.convert(input_path)
        markdown = result.document.export_to_markdown()

        with open(output_md, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"  Saved: {output_md} ({round(time.time() - start, 2)}s)")
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

    def _run_insights_pipeline(self, query: str, markdown: str) -> dict:
        from schema_planner import schema_planner_node
        from extractor import extractor_node
        from router import router_node
        from code_generator import code_generator_node
        from code_executor import executor_node, MAX_RETRIES

        # Step 1 – Schema Planner
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

        # Step 2 – Extractor
        print(f"[2/6] Extractor")
        print(f"      {DIVIDER}")
        statement, transactions = extractor_node(query, schema, markdown)
        print(f"  Transactions extracted : {len(transactions)}")
        for k, v in statement.items():
            print(f"    • {k} : {v}")
        for i, tx in enumerate(transactions, 1):
            print(f"    [{i:02d}] {json.dumps(tx)}")
        print()

        # Step 3 – Router
        print(f"[3/6] Router")
        print(f"      {DIVIDER}")
        decision = router_node(query, statement, transactions)
        print(f"  Needs computation : {decision.needs_computation}")
        print(f"  Reason            : {decision.reason}")
        print()

        exec_result = None
        generated   = None

        if decision.needs_computation:
            # Step 4 – Code Generator
            print(f"[4/6] Code Generator")
            print(f"      {DIVIDER}")
            generated = code_generator_node(query, schema, statement, transactions)
            print(f"  Reasoning : {generated.reasoning}")
            for line in generated.code.splitlines():
                print(f"    {line}")
            print()

            # Step 5 – Executor
            print(f"[5/6] Executor")
            print(f"      {DIVIDER}")
            exec_result = executor_node(query, schema, statement, transactions, generated)
            if exec_result.success:
                print(f"  ✓ Execution successful")
                print(f"  Raw result : {exec_result.result}")
            else:
                print(f"  ✗ Execution failed after {MAX_RETRIES} attempts")
                print(f"  Error : {exec_result.error}")
            print()
        else:
            print(f"[4/6] Code Generator  — SKIPPED")
            print(f"[5/6] Executor        — SKIPPED")
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
            "summary" : narrator.summary,
            "answer"  : narrator.answer,
            "schema"  : {
                "fields"           : [{"name": f.name, "type": f.type, "description": f.description} for f in schema.fields],
                "filter_hint"      : schema.filter_hint,
                "computation_hint" : schema.computation_hint,
                "document_context" : schema.document_context,
            },
            "statement"    : statement,
            "transactions" : transactions[:100],
            "router"       : {"needs_computation": decision.needs_computation, "reason": decision.reason},
            "code_generator" : {
                "reasoning" : generated.reasoning if generated else None,
                "code"      : generated.code      if generated else None,
            },
            "executor" : {
                "success" : exec_result.success if exec_result else None,
                "result"  : exec_result.result  if exec_result else None,
                "error"   : exec_result.error   if exec_result else None,
            },
        }

    def _run_pipeline(self, input_bytes: bytes, filename: str, query: str, mode: str) -> dict:
        import tempfile
        from pathlib import Path

        ext = os.path.splitext(filename)[1].lower()
        tmp_fd, input_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(input_bytes)

        # Convert to markdown
        if ext in {".pdf", ".csv", ".xlsx"}:
            md_path  = self.convert_to_md(input_path)
            markdown = open(md_path, encoding="utf-8").read()
        elif ext == ".md":
            markdown = open(input_path, encoding="utf-8").read()
            md_path  = input_path
        else:
            raise ValueError(f"Unsupported file type: {ext}.")

        os.unlink(input_path)

        doc_name = os.path.splitext(filename)[0]

        if mode == "rag":
            from chunker_v2   import chunk_file
            from embed_v2    import embed_document
            from retriever import retrieve_chunks

            # Chunk
            chunks_dir = Path("/data/chunks") / doc_name
            if chunks_dir.exists() and any(chunks_dir.glob("chunk_*.md")):
                print(f"  [SKIP] Chunks already exist: {chunks_dir}")
            else:
                print(f"  Chunking document...")
                chunk_file(md_path, output_dir=str(chunks_dir))

            # Embed + index
            print(f"  Embedding and indexing into Pinecone...")
            embed_document(doc_name, chunks_dir=str(chunks_dir))

            # Retrieve
            print(f"  Retrieving relevant chunks for query...")
            markdown, chunks_info = retrieve_chunks(query, doc_name)

            if not markdown:
                raise RuntimeError("No relevant chunks found for this query.")

            print(f"  Retrieved {len(chunks_info)} chunk(s):")
            for c in chunks_info:
                print(f"    chunk_{c['chunk_id']} | score: {c['score']} | {c['summary']}")

            # Save merged markdown
            merged_path = Path("/data/merged") / f"{doc_name}-merged.md"
            merged_path.write_text(markdown, encoding="utf-8")
            print(f"  Merged markdown saved: {merged_path}")

        result = self._run_insights_pipeline(query, markdown)
        result["mode"] = mode
        return result

    @modal.method()
    def run_pipeline(self, input_bytes: bytes, filename: str, query: str, mode: str = "direct") -> dict:
        return self._run_pipeline(input_bytes, filename, query, mode)

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        import base64
        file_bytes = base64.b64decode(request.file_b64)
        return self._run_pipeline(
            input_bytes = file_bytes,
            filename    = request.filename,
            query       = request.query,
            mode        = request.mode,
        )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    input_path : str = "",
    query      : str = "",
    mode       : str = "direct",
):
    if not input_path or not query:
        raise ValueError(
            'Provide --input-path, --query, and optionally --mode (rag|direct)\n'
            'e.g.: modal run rag-based-financial-assistant-modal.py --input-path ./statement.pdf --query "What is my total spend?" --mode rag'
        )

    if mode not in {"rag", "direct"}:
        raise ValueError(f"Invalid mode '{mode}'. Use 'rag' or 'direct'.")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File does not exist: {input_path}")

    filename = os.path.basename(input_path)
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    print(f"\n{'=' * 60}")
    print(f"  FINANCIAL DOCUMENT INSIGHTS ENGINE  [{mode.upper()} MODE]")
    print(f"{'=' * 60}")
    print(f"  File  : {filename}")
    print(f"  Query : {query}")
    print(f"  Mode  : {mode}")
    print(f"{'=' * 60}\n")

    engine = FinancialInsightsEngine()
    result = engine.run_pipeline.remote(
        input_bytes = input_bytes,
        filename    = filename,
        query       = query,
        mode        = mode,
    )

    print(f"\n{'=' * 60}")
    print(f"  FINAL ANSWER")
    print(f"{'=' * 60}")
    print(f"  Summary : {result['summary']}")
    print(f"{'=' * 60}")
    print(f"\n{result['answer']}")
    print(f"\n{'=' * 60}\n")