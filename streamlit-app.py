"""
Streamlit app for the Financial Document Insights Engine (v2)
Supports RAG and Direct mode via toggle.
"""

import os
import base64
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("INFERENCE_ENDPOINT_V2", "")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Jupiter · Financial Insights",
    page_icon="◈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0e0e0f;
    color: #e8e3db;
    font-size: 18px;
}
.stApp { background-color: #0e0e0f; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1100px; padding-left: 3rem; padding-right: 3rem; }

.jupiter-header { text-align: center; margin-bottom: 3rem; }
.jupiter-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.8rem; font-weight: 400;
    color: #e8e3db; letter-spacing: -0.02em; margin: 0; line-height: 1;
}
.jupiter-header h1 span { font-style: italic; color: #c9a96e; }
.jupiter-header p {
    font-size: 0.95rem; color: #5a5a5a;
    letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.6rem;
}

.thin-divider { border: none; border-top: 1px solid #1e1e1f; margin: 2rem 0; }

[data-testid="stFileUploader"] {
    background: #131314; border: 1px solid #2a2a2b;
    border-radius: 2px; padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #c9a96e; }
[data-testid="stFileUploader"] label {
    color: #5a5a5a !important; font-size: 0.78rem !important;
    letter-spacing: 0.1em; text-transform: uppercase;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #131314 !important; border: 1px solid #2a2a2b !important;
    border-radius: 2px !important; color: #e8e3db !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important; padding: 1rem !important;
    resize: none !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus { border-color: #c9a96e !important; box-shadow: none !important; }
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    color: #5a5a5a !important; font-size: 0.9rem !important;
    letter-spacing: 0.1em; text-transform: uppercase;
}

.stButton > button {
    background: #c9a96e !important; color: #0e0e0f !important;
    border: none !important; border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important; font-weight: 500 !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    padding: 0.65rem 2rem !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stProgress"] > div > div { background: #c9a96e !important; }

/* ── Mode toggle ── */
.mode-toggle-wrap {
    display: flex; align-items: center; gap: 1rem;
    background: #131314; border: 1px solid #2a2a2b;
    border-radius: 2px; padding: 0.85rem 1.2rem;
    margin-bottom: 1.25rem;
}
.mode-label {
    font-size: 0.78rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: #5a5a5a; white-space: nowrap;
}
.mode-badge {
    display: inline-block; padding: 0.25rem 0.85rem;
    border-radius: 2px; font-size: 0.82rem;
    letter-spacing: 0.12em; text-transform: uppercase; font-weight: 500;
}
.mode-badge-rag    { background: #1a2a1a; color: #6abf6a; border: 1px solid #2a4a2a; }
.mode-badge-direct { background: #1a1a2a; color: #6a9abf; border: 1px solid #2a2a4a; }
.mode-hint {
    font-size: 0.82rem; color: #3a3a3b;
    letter-spacing: 0.05em; margin-left: auto;
}

/* toggle switch override */
[data-testid="stToggle"] { margin: 0 !important; }
[data-testid="stToggle"] label { display: none !important; }

.result-card {
    background: #131314; border: 1px solid #2a2a2b;
    border-left: 3px solid #c9a96e; border-radius: 2px;
    padding: 1.5rem 1.8rem; margin-bottom: 1rem;
}
.result-label {
    font-size: 0.9rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: #5a5a5a; margin-bottom: 0.5rem;
}
.result-summary {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; color: #c9a96e; line-height: 1.3;
}
.result-answer { font-size: 1.05rem; color: #b0ab9e; line-height: 1.9; white-space: pre-wrap; }

.mode-result-pill {
    display: inline-block; padding: 0.3rem 1rem;
    border-radius: 2px; font-size: 0.82rem;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 1.25rem;
}
.mode-result-rag    { background: #1a2a1a; color: #6abf6a; border: 1px solid #2a4a2a; }
.mode-result-direct { background: #1a1a2a; color: #6a9abf; border: 1px solid #2a2a4a; }

.meta-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1rem; }
.meta-pill {
    background: #1a1a1b; border: 1px solid #2a2a2b; border-radius: 2px;
    padding: 0.4rem 1rem; font-size: 0.88rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: #5a5a5a;
}
.meta-pill span { color: #e8e3db; }

.error-box {
    background: #1a0f0f; border: 1px solid #3d1f1f;
    border-left: 3px solid #c0392b; border-radius: 2px;
    padding: 1rem 1.4rem; font-size: 1rem; color: #e07070;
}

/* expander overrides */
[data-testid="stExpander"] {
    background: #131314 !important;
    border: 1px solid #2a2a2b !important;
    border-radius: 2px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 1rem !important;
    letter-spacing: 0.08em !important;
    color: #7a7a7a !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: #c9a96e !important; }

.code-block {
    background: #0a0a0b; border: 1px solid #1e1e1f;
    border-radius: 2px; padding: 1rem 1.2rem;
    font-size: 0.95rem; color: #7abfb0;
    white-space: pre-wrap; line-height: 1.7;
    overflow-x: auto;
}
.field-row {
    display: flex; gap: 1rem; padding: 0.5rem 0;
    border-bottom: 1px solid #1a1a1b; font-size: 0.95rem;
}
.field-name { color: #c9a96e; min-width: 140px; }
.field-type { color: #5a5a5a; min-width: 60px; }
.field-desc { color: #7a7a7a; }
.kv-row { display: flex; gap: 1rem; padding: 0.5rem 0; font-size: 0.95rem; }
.kv-key { color: #7a7a7a; min-width: 180px; }
.kv-val { color: #e8e3db; }
.tag {
    display: inline-block; padding: 0.2rem 0.6rem;
    border-radius: 2px; font-size: 0.88rem;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.tag-yes  { background: #0f2a0f; color: #6abf6a; border: 1px solid #1f4a1f; }
.tag-no   { background: #1a1a0f; color: #bfbf6a; border: 1px solid #3a3a1f; }
.tag-ok   { background: #0f2a0f; color: #6abf6a; border: 1px solid #1f4a1f; }
.tag-fail { background: #2a0f0f; color: #bf6a6a; border: 1px solid #4a1f1f; }
.tag-skip { background: #1a1a1b; color: #5a5a5a; border: 1px solid #2a2a2b; }

/* ── RAG chunks expander ── */
.chunk-row {
    display: flex; gap: 1rem; padding: 0.6rem 0;
    border-bottom: 1px solid #1a1a1b; font-size: 0.92rem; align-items: baseline;
}
.chunk-id    { color: #c9a96e; min-width: 80px; }
.chunk-score { color: #5a5a5a; min-width: 70px; }
.chunk-summary { color: #7a7a7a; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="jupiter-header">
    <h1>◈ <span>Jupiter</span></h1>
    <p>Financial Document Insights Engine</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

# ── Mode toggle ───────────────────────────────────────────────────────────────

col_label, col_toggle, col_badge, col_hint = st.columns([2, 0.6, 1.2, 3])

with col_label:
    st.markdown("<div style='padding-top:0.45rem;font-size:0.78rem;letter-spacing:0.15em;text-transform:uppercase;color:#5a5a5a'>Mode</div>", unsafe_allow_html=True)

with col_toggle:
    use_rag = st.toggle("RAG", value=False, label_visibility="collapsed")

mode = "rag" if use_rag else "direct"

with col_badge:
    if use_rag:
        st.markdown(
            '<div style="padding-top:0.35rem"><span class="mode-badge mode-badge-rag">RAG</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="padding-top:0.35rem"><span class="mode-badge mode-badge-direct">Direct</span></div>', unsafe_allow_html=True)

with col_hint:
    if use_rag:
        st.markdown("<div style='padding-top:0.45rem;font-size:0.78rem;color:#3a3a3b;letter-spacing:0.05em'>chunk → embed → retrieve → analyse</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding-top:0.45rem;font-size:0.78rem;color:#3a3a3b;letter-spacing:0.05em'>full document → analyse</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

# ── Upload & query ────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Statement", type=["pdf", "csv", "xlsx"])
query = st.text_area(
    "Query", placeholder='e.g. "What is my total spend this month?"', height=100)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

run_clicked = st.button(
    "Analyse →", disabled=not uploaded_file or not query.strip())

# ── Run ───────────────────────────────────────────────────────────────────────

if run_clicked and uploaded_file and query.strip():
    st.session_state.result = None
    st.session_state.error = None

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    file_bytes = uploaded_file.read()
    file_b64 = base64.b64encode(file_bytes).decode("utf-8")
    payload = {
        "file_b64": file_b64,
        "filename": uploaded_file.name,
        "query": query.strip(),
        "mode": mode,
    }

    spinner_msg = "Chunking, embedding and retrieving..." if mode == "rag" else "Analysing document..."

    result_holder = {}
    with st.spinner(spinner_msg):
        try:
            resp = requests.post(ENDPOINT, json=payload, timeout=600)
            if resp.status_code == 200:
                result_holder["data"] = resp.json()
            else:
                result_holder["error"] = f"Endpoint returned {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            result_holder["error"] = str(e)

    if "error" in result_holder:
        st.session_state.error = result_holder["error"]
    else:
        st.session_state.result = result_holder["data"]

# ── Render result ─────────────────────────────────────────────────────────────

if st.session_state.result:
    r = st.session_state.result
    result_mode = r.get("mode", "direct")

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    # Mode badge on result
    if result_mode == "rag":
        st.markdown(
            '<span class="mode-result-pill mode-result-rag">◈ RAG mode</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="mode-result-pill mode-result-direct">◈ Direct mode</span>', unsafe_allow_html=True)

    # ── Final answer ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Summary</div>
        <div class="result-summary">{r.get('summary', '')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Answer</div>
        <div class="result-answer">{r.get('answer', '')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.92rem;letter-spacing:0.2em;text-transform:uppercase;color:#5a5a5a;margin-bottom:0.75rem'>Pipeline Trace</p>", unsafe_allow_html=True)

    # ── Step 1: Schema Planner ────────────────────────────────────────────────
    schema = r.get("schema", {})
    fields = schema.get("fields", [])
    with st.expander(f"1 / 6 · Schema Planner — {len(fields)} fields planned"):
        fields_html = "".join([
            f'<div class="field-row"><span class="field-name">{f["name"]}</span><span class="field-type">{f["type"]}</span><span class="field-desc">{f["description"]}</span></div>'
            for f in fields
        ])
        st.markdown(
            f'<div style="margin-bottom:1rem">{fields_html}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kv-row"><span class="kv-key">Filter hint</span><span class="kv-val">{schema.get('filter_hint', '—')}</span></div>
        <div class="kv-row"><span class="kv-key">Computation hint</span><span class="kv-val">{schema.get('computation_hint', '—')}</span></div>
        <div class="kv-row"><span class="kv-key">Document context</span><span class="kv-val">{schema.get('document_context', '—')}</span></div>
        """, unsafe_allow_html=True)

    # ── Step 2: Extractor ─────────────────────────────────────────────────────
    statement = r.get("statement", {})
    transactions = r.get("transactions", [])
    with st.expander(f"2 / 6 · Extractor — {len(transactions)} transactions"):
        if statement:
            st.markdown(
                "<div class='result-label' style='margin-bottom:0.5rem'>Statement</div>", unsafe_allow_html=True)
            kv_html = "".join([
                f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>'
                for k, v in statement.items()
            ])
            st.markdown(
                f'<div style="margin-bottom:1rem">{kv_html}</div>', unsafe_allow_html=True)
        if transactions:
            st.markdown(
                "<div class='result-label' style='margin-bottom:0.5rem'>Transactions</div>", unsafe_allow_html=True)
            import pandas as pd
            st.dataframe(
                pd.DataFrame(transactions),
                use_container_width=True,
                hide_index=True,
            )

    # ── Step 3: Router ────────────────────────────────────────────────────────
    router = r.get("router", {})
    needs = router.get("needs_computation", False)
    tag = '<span class="tag tag-yes">yes</span>' if needs else '<span class="tag tag-no">no</span>'
    with st.expander(f"3 / 6 · Router — computation needed: {'yes' if needs else 'no'}"):
        st.markdown(f"""
        <div class="kv-row"><span class="kv-key">Needs computation</span><span class="kv-val">{tag}</span></div>
        <div class="kv-row"><span class="kv-key">Reason</span><span class="kv-val">{router.get('reason', '—')}</span></div>
        """, unsafe_allow_html=True)

    # ── Step 4: Code Generator ────────────────────────────────────────────────
    cg = r.get("code_generator", {})
    code = cg.get("code")
    with st.expander("4 / 6 · Code Generator" + (" — skipped" if not code else "")):
        if code:
            st.markdown(
                f'<div class="kv-row"><span class="kv-key">Reasoning</span><span class="kv-val">{cg.get("reasoning", "—")}</span></div>', unsafe_allow_html=True)
            st.markdown("<div style='height:0.75rem'></div>",
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="code-block">{code}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<span class="tag tag-skip">skipped — no computation needed</span>', unsafe_allow_html=True)

    # ── Step 5: Executor ──────────────────────────────────────────────────────
    executor = r.get("executor", {})
    success = executor.get("success")
    with st.expander("5 / 6 · Executor" + (" — skipped" if success is None else (" — ✓ success" if success else " — ✗ failed"))):
        if success is None:
            st.markdown(
                '<span class="tag tag-skip">skipped — no computation needed</span>', unsafe_allow_html=True)
        else:
            status_tag = '<span class="tag tag-ok">success</span>' if success else '<span class="tag tag-fail">failed</span>'
            st.markdown(
                f'<div class="kv-row"><span class="kv-key">Status</span><span class="kv-val">{status_tag}</span></div>', unsafe_allow_html=True)
            if executor.get("result"):
                st.markdown(
                    f'<div class="kv-row"><span class="kv-key">Result</span><span class="kv-val">{executor["result"]}</span></div>', unsafe_allow_html=True)
            if executor.get("error"):
                st.markdown(
                    f'<div class="kv-row"><span class="kv-key">Error</span><span class="kv-val" style="color:#e07070">{executor["error"]}</span></div>', unsafe_allow_html=True)

if st.session_state.error:
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="error-box">✗ &nbsp;{st.session_state.error}</div>', unsafe_allow_html=True)
