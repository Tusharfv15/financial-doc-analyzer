"""
Retriever
----------
Given a user query and source PDF name:
1. Embeds the query using text-embedding-3-small
2. Queries Pinecone with vector similarity + optional metadata filter
3. Returns top-K relevant chunks merged into a single markdown string

Usage:
    python retriever.py <pdf_name> "<query>"
    python retriever.py 379281148-Sample-Credit-Card-Bill "Show me all EMI transactions"

Environment variables required:
    OPENAI_API_KEY
    PINECONE_API_KEY
    PINECONE_INDEX_NAME  (default: jupiter-chunks)
"""

import os
import sys
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(override=True)
OPENAI_CLIENT  = OpenAI()
PINECONE_CLIENT = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME     = os.environ.get("PINECONE_INDEX_NAME", "jupiter-chunks")
EMBED_MODEL    = "text-embedding-3-small"
TOP_K          = 3


# ── Embed query ────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    response = OPENAI_CLIENT.embeddings.create(
        model = EMBED_MODEL,
        input = query,
    )
    return response.data[0].embedding


# ── Retrieve chunks ────────────────────────────────────────────────────────────

def retrieve_chunks(
    query      : str,
    source_pdf : str,
    top_k      : int = TOP_K,
    chunk_types: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """
    Retrieve top-K relevant chunks for a query from Pinecone.

    Args:
        query       : user query string
        source_pdf  : PDF name to filter by (only retrieve chunks from this doc)
        top_k       : number of chunks to retrieve
        chunk_types : optional list of chunk_type to filter e.g. ['transaction_table', 'summary']

    Returns:
        (merged_markdown, list of chunk metadata dicts)
    """
    index     = PINECONE_CLIENT.Index(INDEX_NAME)
    embedding = embed_query(query)

    # Build metadata filter
    # Always filter by source_pdf and only retrieve insight-relevant chunks
    metadata_filter = {
        "source_pdf"               : {"$eq"  : source_pdf},
        "is_relevant_for_insights" : {"$eq"  : True},
    }

    if chunk_types:
        metadata_filter["chunk_types"] = {"$in": chunk_types}

    results = index.query(
        vector          = embedding,
        top_k           = top_k,
        filter          = metadata_filter,
        include_metadata= True,
    )

    if not results.matches: # type: ignore
        return "", []

    # Sort by chunk_id to preserve document order
    matches = sorted(results.matches, key=lambda m: m.metadata.get("chunk_id", 0)) # type: ignore

    chunks_info = []
    texts       = []

    for match in matches:
        meta = match.metadata
        chunks_info.append({
            "chunk_id"   : meta.get("chunk_id"),
            "chunk_types": meta.get("chunk_types"),
            "summary"    : meta.get("summary"),
            "date_range" : meta.get("date_range"),
            "contains"   : meta.get("contains"),
            "score"      : round(match.score, 4),
        })
        texts.append(meta.get("text", ""))

    merged_markdown = "\n\n---\n\n".join(texts)

    return merged_markdown, chunks_info


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python retriever.py <pdf_name> "<query>"')
        sys.exit(1)

    pdf_name = sys.argv[1]
    query    = sys.argv[2]

    print(f"\n{'=' * 60}")
    print(f"  Retriever")
    print(f"{'=' * 60}")
    print(f"  Query  : {query}")
    print(f"  Source : {pdf_name}")
    print(f"{'=' * 60}\n")

    print(f"Retrieving top-{TOP_K} chunks...")
    markdown, chunks_info = retrieve_chunks(query, pdf_name)

    print(f"\n── Retrieved Chunks ({len(chunks_info)}) ─────────────────────────────────────")
    for c in chunks_info:
        print(f"  chunk_{c['chunk_id']} | types: {c['chunk_types']} | score: {c['score']}")
        print(f"    summary    : {c['summary']}")
        print(f"    date_range : {c['date_range']}")
        print(f"    contains   : {c['contains']}")
        print()

    print(f"── Merged Markdown ({len(markdown)} chars) ────────────────────────────────────")
    print(markdown[:500] + "..." if len(markdown) > 500 else markdown)