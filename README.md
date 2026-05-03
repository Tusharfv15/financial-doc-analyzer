# Jupiter Assignment — Financial Document Insights Engine

A pipeline that answers natural language queries over financial PDFs.

---

## Usage

```bash
# Direct mode (default)
python modal-financial_assistant-rag-pipeline.py docs/statement.pdf "Total amount of transactions done Via UPI"

# RAG mode
python modal-financial_assistant-rag-pipeline.py docs/statement.pdf "Detailed Report of scheme performance discussed in that of Annexure 1" --rag
```

---

## Modes

### Direct (default)
Best for short documents (1–5 pages). The full markdown is passed straight to the insights pipeline.

```
PDF → Docling OCR → Markdown → Insights Pipeline → Answer
```

### RAG (`--rag`)
Best for large multi-page documents. Indexes the document into Pinecone, retrieves only the chunks relevant to the query, then runs insights on those chunks.

```
PDF → Docling OCR → Markdown → Chunker → Embed → Pinecone
                                                      ↓
                                              Retrieve top-K chunks
                                              Save to merged_markdown/
                                                      ↓
                                           Insights Pipeline → Answer
```

> Both modes skip OCR and chunking if outputs already exist on disk.

---

## Insights Pipeline

Shared by both modes. Takes a markdown string + query and returns a natural language answer.

```
Query + Markdown
      ↓
Schema Planner   — decides what fields and transactions to extract
      ↓
Extractor        — extracts a statement summary dict + list of transactions
      ↓
Router           — decides if numeric computation is needed
      ↓
Code Generator   — writes Python code to compute the result  (skipped if not needed)
      ↓
Executor         — runs the code in a sandbox, retries up to 2x on failure
      ↓
Narrator         — formats everything into a clear natural language answer
```

> The LLM never does math directly — it only extracts, plans, generates code, and narrates. All computation goes through the sandboxed executor.

---
## Architecture
<img width="5460" height="6127" alt="Untitled-2025-09-10-2053" src="https://github.com/user-attachments/assets/990f661b-57cb-4ec9-aeec-4d23120e4ab5" />



## Project Structure

```
modal-financial-assistant-rag-pipeline.py   # entry point — orchestrates both modes
chunker_v2.py                               # splits markdown into overlapping chunks at ## headings
schema_planner.py        # plans extraction schema from query + document
extractor.py             # extracts statement summary + transactions
router.py                # decides if computation is needed
code_generator.py        # generates Python code for numeric computation
code_executor.py         # sandboxed exec with retry loop
narrator.py              # produces final natural language answer
embed_v2.py                 # embeds chunks and upserts to Pinecone
retriever.py             # retrieves top-K relevant chunks from Pinecone

docling_md/              # full markdown per PDF (OCR output)
doc_chunks/              # chunks per PDF
merged_markdown/         # retrieved + merged chunks used for each query
```

---

## Deployment

### Backend — Modal
Docling OCR runs on a Modal serverless GPU endpoint. The `modal-financial-assistant-rag-pipeline.py ` script is deployed as a Modal function, so OCR happens on cloud GPUs without any local setup. The main pipeline calls this endpoint and receives the markdown output.
<img width="1550" height="152" alt="image" src="https://github.com/user-attachments/assets/de84a2bb-e1d0-41e7-9cd0-eedea99983a2" />


### Frontend — Streamlit
The user interface is a Streamlit app. Users upload a PDF, type a query, choose a mode (Direct or RAG), and see the final answer along with intermediate outputs (extracted transactions, generated code, narrator response).

### UI
<img width="1335" height="629" alt="image" src="https://github.com/user-attachments/assets/e91aa802-ba42-4250-b9f1-637f6f7d6ed6" />

## Summary
<img width="1203" height="832" alt="image" src="https://github.com/user-attachments/assets/5514c788-7a29-4722-8ace-360a50381664" />

## Extracted Transactions from the document
<img width="1284" height="864" alt="image" src="https://github.com/user-attachments/assets/0da5c5e6-28ae-4cf8-972e-f8178da4780a" />



