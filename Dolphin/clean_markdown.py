"""
Layer 1: Generic OCR Noise Removal for Bank Statements
-------------------------------------------------------
Cleans raw OCR markdown output before passing to LLM (Layer 2).
Statement-agnostic — works across ICICI, HDFC, SBI, Axis, Amex, etc.
"""

import re


def clean_ocr_statement(raw_text: str) -> str:
    """
    Main entry point. Runs all cleaning steps in order.
    Returns cleaned text ready for LLM ingestion.
    """
    text = raw_text

    text = remove_image_references(text)
    text = remove_page_metadata(text)
    text = fix_currency_symbols(text)
    text = remove_horizontal_rules(text)
    text = clean_table_artifacts(text)
    text = fix_whitespace(text)
    text = remove_boilerplate_blocks(text)

    return text.strip()


# ── Step 1: Remove image/figure references ────────────────────────────────────

def remove_image_references(text: str) -> str:
    """Remove markdown image tags: ![Figure](...) or ![...](...)"""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    return text


# ── Step 2: Remove page-level metadata noise ──────────────────────────────────

PAGE_METADATA_PATTERNS = [
    r'Page\s+\d+\s+of\s+\d+',                          # Page 1 of 4
    r'Invoke\s+No[:\s]+[\w]+',                          # Invoke No: 157415...
    r'CIN\s+No[.\s:]+[\w\d]+',                          # CIN No. L65190GJ...
    r'\d+/\d+\s*$',                                     # 1/1 at end of line
    r'^---+$',                                           # horizontal dividers
    r'^\s*1/1\s*$',                                     # standalone 1/1
]

def remove_page_metadata(text: str) -> str:
    for pattern in PAGE_METADATA_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    return text


# ── Step 3: Fix currency symbol corruption ────────────────────────────────────

def fix_currency_symbols(text: str) -> str:
    """
    OCR often garbles ₹ into backtick, backtick+grave, Rs., INR etc.
    Normalize all to ₹ for consistency.
    """
    # Backtick used as rupee (common in markdown OCR output)
    text = re.sub(r'`(\d)', r'₹\1', text)

    # Rs. or Rs variants
    text = re.sub(r'\bRs\.?\s*', '₹', text, flags=re.IGNORECASE)

    # INR followed by amount
    text = re.sub(r'\bINR\s*', '₹', text, flags=re.IGNORECASE)

    # Garbled unicode rupee (common in bad PDF→OCR pipelines)
    text = text.replace('\x60', '₹')  # grave accent

    return text


# ── Step 4: Remove horizontal rules ───────────────────────────────────────────

def remove_horizontal_rules(text: str) -> str:
    """Remove markdown --- or === dividers"""
    text = re.sub(r'^\s*[-=]{3,}\s*$', '', text, flags=re.MULTILINE)
    return text


# ── Step 5: Clean table artifacts ─────────────────────────────────────────────

def clean_table_artifacts(text: str) -> str:
    """
    - Remove orphan card number rows (standalone masked numbers like 4375XXXX...)
    - Flag/remove rows that are clearly OCR-broken (e.g. date 13/17/2020)
    - Normalize CR/DR markers to consistent format
    """
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        # Remove orphan card number fragments (standalone masked PANs)
        if re.match(r'^\s*\d{4}[X\d]{8,}\d{4}\s*$', line):
            continue

        # Remove lines that are just series numbers or reference codes alone
        if re.match(r'^\s*[A-Z0-9]{10,}\s*$', line) and '|' not in line:
            continue

        # Normalize amount suffixes: "1,000.00 CR" → "1,000.00 CR" (keep but standardize)
        line = re.sub(r'\bC\.R\b', 'CR', line)
        line = re.sub(r'\bD\.R\b', 'DR', line)

        # Flag suspicious dates (month > 12) with a comment — don't silently drop
        if re.search(r'\b\d{2}/(?:1[3-9]|[2-9]\d)/\d{4}\b', line):
            line = line + '  # [OCR_DATE_ERROR?]'

        cleaned.append(line)

    return '\n'.join(cleaned)


# ── Step 6: Fix whitespace ────────────────────────────────────────────────────

def fix_whitespace(text: str) -> str:
    """
    - Collapse 3+ consecutive blank lines to 2
    - Strip trailing whitespace per line
    """
    # Strip trailing spaces per line
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


# ── Step 7: Remove boilerplate blocks ─────────────────────────────────────────

BOILERPLATE_MARKERS = [
    "MOST IMPORTANT TERMS AND CONDITIONS",
    "GREAT OFFERS ON YOUR CARD",
    "IMPORTANT MESSAGES",
    "Safe Banking Tips",
    "To get the complete version of Credit Cards",
    "Making only minimum payment every month",
    "Please pay your Credit Card outstanding",
    "For any clarification or more information",
    "Mark-up fee and corresponding GST",
    "Payment through UPI is subject to",
    "SMS Mobile to",
    "Download the iMobile app",
    "All communications are being sent",
    "To update email ID",
    "To update mobile number",
    "This is an authenticated intimation",
    "TMC apply",
    "Offer Valid till",
    "For more details, visit www.icicibank.com/creditcardoffers",
]

def remove_boilerplate_blocks(text: str) -> str:
    """
    Remove blocks of text that start with known boilerplate markers.
    Removes the marker line and the paragraph following it.
    """
    lines = text.split('\n')
    cleaned = []
    skip_until_blank = False

    for line in lines:
        # Check if this line starts a boilerplate block
        is_boilerplate = any(marker.lower() in line.lower() for marker in BOILERPLATE_MARKERS)

        if is_boilerplate:
            skip_until_blank = True
            continue

        if skip_until_blank:
            if line.strip() == '':
                skip_until_blank = False
            continue

        cleaned.append(line)

    return '\n'.join(cleaned)


# ── Utility: diff preview ──────────────────────────────────────────────────────

def preview_diff(original: str, cleaned: str) -> None:
    """Print a simple before/after line count and char count summary."""
    orig_lines = len(original.splitlines())
    clean_lines = len(cleaned.splitlines())
    print(f"Lines : {orig_lines} → {clean_lines} ({orig_lines - clean_lines} removed)")
    print(f"Chars : {len(original)} → {len(cleaned)} ({len(original) - len(cleaned)} removed)")
    print(f"Noise removed: {((len(original) - len(cleaned)) / len(original) * 100):.1f}%")


# ── CLI / quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python layer1_cleaner.py <path_to_ocr_markdown.md>")
        print("\nRunning on sample ICICI statement for demo...")

        sample = open("/mnt/user-data/uploads/753706830-ICICI-CREDIT-CARD-STATEMENT.md").read()
        cleaned = clean_ocr_statement(sample)

        print("\n" + "="*60)
        print("CLEANED OUTPUT:")
        print("="*60)
        print(cleaned)
        print("\n" + "="*60)
        print("STATS:")
        preview_diff(sample, cleaned)

    else:
        path = sys.argv[1]
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()

        cleaned = clean_ocr_statement(raw)

        out_path = path.replace('.md', '_cleaned.md')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        print(f"Cleaned output written to: {out_path}")
        preview_diff(raw, cleaned)