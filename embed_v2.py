"""
Stage 1.5 — Metadata Generation + Pinecone Embedding (v2)
----------------------------------------------------------
Accepts an optional chunks_dir parameter so it can read chunks
from any directory (e.g. Modal Volume at /data/chunks/<doc_name>)
instead of the hardcoded local doc_chunks/<pdf_name>/ path.
"""

import os
import re
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

OPENAI_CLIENT = OpenAI()
PINECONE_CLIENT = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "jupiter-chunks")
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
EMBED_DIMENSIONS = 1536
CHUNKS_DIR = Path("doc_chunks")


# ── Pydantic schema for metadata ───────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    chunk_types: list[str] = Field(description=(
        "Top 1-3 content types present in this chunk — pick only the most dominant ones. "
        "Infer from content. Examples: "
        "'transaction_table', 'account_summary', 'emi_schedule', "
        "'scheme_performance', 'balance_sheet', 'auditors_report', "
        "'corporate_governance', 'charges_and_fees', 'credit_summary', "
        "'investment_details', 'regulatory_disclosure', 'promotional_content'. "
        "Use snake_case. Never list more than 3."
    ))
    summary: str = Field(description=(
        "1-2 sentence description of what this chunk is about. "
        "Mention the document section, key entities, and time period. "
        "Do not include specific numeric values. "
        "Example: 'Mutual fund scheme performance summary for a financial year with auditor opinion.' "
        "Example: 'Monthly transaction table with UPI payments and retail purchases, plus tax charges.'"
    ))
    date_range: str = Field(description=(
        "Date range of data in this chunk. "
        "Could be a statement period, transaction dates, financial year, or report date. "
        "Examples: 'July 2020', 'FY 2023-24', 'June 22 2024 to July 22 2024'. "
        "Use 'unknown' only if absolutely no dates present."
    ))
    contains: list[str] = Field(description=(
        "Entity names only — merchants, scheme names, transaction types, financial terms, institutions. "
        "Do NOT include numeric values, percentages, or key-value pairs. "
        "Examples: ['Reliance Digital', 'UPI Payment', 'HDFC Index Fund', 'S&P BSE SENSEX', 'SGST', 'EMI']. "
        "Max 8 items. Only entity names that are clearly present."
    ))
    is_relevant_for_insights: bool = Field(description=(
        "True if this chunk contains actual financial data useful for insights — "
        "transactions, balances, NAV, returns, charges, EMI, summaries. "
        "False if this chunk is primarily legal boilerplate, promotional content, "
        "auditor disclaimers, FATCA/CRS notices, or acknowledgements."
    ))


# ── Step 1: Generate metadata ──────────────────────────────────────────────────

def generate_metadata(chunk_text: str) -> ChunkMetadata:
    response = OPENAI_CLIENT.responses.parse(
        model=LLM_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a financial document metadata extractor. "
                    "You will be given a chunk of markdown content from a financial document. "
                    "Your job is to extract structured metadata about the chunk to aid retrieval. "
                    "Be precise — only extract what is explicitly present in the chunk. "
                    "Do not infer or hallucinate."
                )
            },
            {
                "role": "user",
                "content": f"Chunk content:\n\n{chunk_text}"
            }
        ],
        text_format=ChunkMetadata,
    )
    return response.output_parsed  # type: ignore


# ── Step 2: Embed chunk text ───────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    response = OPENAI_CLIENT.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding


# ── Step 3: Ensure Pinecone index exists ───────────────────────────────────────

def ensure_index():
    existing = [i.name for i in PINECONE_CLIENT.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"  Creating Pinecone index '{INDEX_NAME}'...")
        PINECONE_CLIENT.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"  Index '{INDEX_NAME}' created.")
    else:
        print(f"  Index '{INDEX_NAME}' already exists.")
    return PINECONE_CLIENT.Index(INDEX_NAME)


# ── Step 4: Upsert to Pinecone ─────────────────────────────────────────────────

def upsert_chunk(
    index: object,
    vector_id: str,
    embedding: list[float],
    metadata: ChunkMetadata,
    chunk_text: str,
    source_pdf: str,
    chunk_id: int,
    has_table: bool,
):
    index.upsert(vectors=[{  # type: ignore
        "id": vector_id,
        "values": embedding,
        "metadata": {
            "source_pdf": source_pdf,
            "chunk_id": chunk_id,
            "chunk_types": ", ".join(metadata.chunk_types),
            "summary": metadata.summary,
            "date_range": metadata.date_range,
            "contains": ", ".join(metadata.contains),
            "has_table": has_table,
            "is_relevant_for_insights": metadata.is_relevant_for_insights,
            "text": chunk_text,
        }
    }])


# ── Helpers ────────────────────────────────────────────────────────────────────

def has_table(text: str) -> bool:
    return any(re.match(r"^\|", l.strip()) for l in text.splitlines())


# ── Main pipeline ──────────────────────────────────────────────────────────────

# type: ignore # ← fixed signature
def embed_document(pdf_name: str, chunks_dir: str = None): # type: ignore
    chunks_path = Path(chunks_dir) if chunks_dir else CHUNKS_DIR / pdf_name

    if not chunks_path.exists():
        print(f"[ERROR] No chunks found for '{pdf_name}' at {chunks_path}")
        sys.exit(1)

    chunk_files = sorted(chunks_path.glob("chunk_*.md"))
    if not chunk_files:
        print(f"[ERROR] No chunk files found in {chunks_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Stage 1.5 — Embed & Index")
    print(f"{'=' * 60}")
    print(f"  Document  : {pdf_name}")
    print(f"  Chunks    : {len(chunk_files)}")
    print(f"  Directory : {chunks_path}")
    print(f"{'=' * 60}\n")

    # Ensure index
    print("[1/3] Ensuring Pinecone index...")
    index = ensure_index()
    print()

    # Process each chunk
    print(
        f"[2/3] Generating metadata + embeddings for {len(chunk_files)} chunks...")
    results = []
    for chunk_file in chunk_files:
        chunk_id = int(chunk_file.stem.split("_")[1])
        chunk_text = chunk_file.read_text(encoding="utf-8")
        vector_id = f"{pdf_name}__chunk_{chunk_id}"

        print(f"  Processing {chunk_file.name}...")

        metadata = generate_metadata(chunk_text)
        print(f"    chunk_types  : {metadata.chunk_types}")
        print(f"    summary      : {metadata.summary}")
        print(f"    date_range   : {metadata.date_range}")
        print(f"    contains     : {metadata.contains}")
        print(f"    is_relevant  : {metadata.is_relevant_for_insights}")

        embedding = embed_text(chunk_text)
        print(f"    embedding    : {len(embedding)} dims")

        results.append({
            "vector_id": vector_id,
            "embedding": embedding,
            "metadata": metadata,
            "chunk_text": chunk_text,
            "chunk_id": chunk_id,
            "has_table": has_table(chunk_text),
        })
        print()

    # Upsert all to Pinecone
    print(f"[3/3] Upserting {len(results)} vectors to Pinecone...")
    for r in results:
        upsert_chunk(
            index=index,
            vector_id=r["vector_id"],
            embedding=r["embedding"],
            metadata=r["metadata"],
            chunk_text=r["chunk_text"],
            source_pdf=pdf_name,
            chunk_id=r["chunk_id"],
            has_table=r["has_table"],
        )
        print(f"  ✓ Upserted {r['vector_id']}")

    print(f"\n{'=' * 60}")
    print(f"  ✓ Done — {len(results)} chunks indexed in Pinecone")
    print(f"  Index : {INDEX_NAME}")
    print(f"{'=' * 60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embed_v2.py <pdf_name> [chunks_dir]")
        print("Example: python embed_v2.py 379281148-Sample-Credit-Card-Bill")
        print("Example: python embed_v2.py 379281148-Sample-Credit-Card-Bill /data/chunks/379281148-Sample-Credit-Card-Bill")
        sys.exit(1)

    chunks_dir_arg = sys.argv[2] if len(sys.argv) > 2 else None
    embed_document(sys.argv[1], chunks_dir=chunks_dir_arg)  # type: ignore
