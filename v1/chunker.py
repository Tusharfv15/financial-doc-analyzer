import re
import os
import sys


MAX_TOKENS = 4500


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def is_table_line(line: str) -> bool:
    return bool(re.match(r'^\|', line.strip()))


def is_heading(line: str) -> bool:
    return line.startswith('##')


def extract_sections(text: str) -> list[str]:
    """
    Split the document into sections at every ## heading.
    Each section = the ## heading line + all content until the next ## heading.
    Content is preserved exactly as in source.
    """
    lines = text.splitlines()
    sections = []
    current_lines = []

    for line in lines:
        if is_heading(line) and current_lines:
            # Save what we have, start a new section
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(section_text)
            current_lines = [line]
        else:
            current_lines.append(line)

    # Last section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(section_text)

    return sections


def has_table(text: str) -> bool:
    return any(re.match(r'^\|', l.strip()) for l in text.splitlines())


def chunk_sections(sections: list[str], max_tokens: int = MAX_TOKENS) -> list[dict]:
    chunks = []
    current_sections = []
    current_tokens = 0
    chunk_index = 0
    last_section = None  # for overlap

    def flush():
        nonlocal current_sections, current_tokens, chunk_index
        if current_sections:
            chunks.append({
                "chunk_id": chunk_index,
                "text": "\n\n".join(current_sections),
                "tokens": current_tokens,
            })
            chunk_index += 1
            current_sections = []
            current_tokens = 0

    for section in sections:
        stokens = estimate_tokens(section)

        if current_tokens + stokens > max_tokens and current_sections:
            flush()
            # Overlap: prepend last non-table section (excluded from token budget)
            if last_section and not has_table(last_section):
                current_sections.append(last_section)

        current_sections.append(section)
        current_tokens += stokens
        last_section = section

    flush()
    return chunks


def chunk_file(filepath: str):
    if not os.path.exists(filepath):
        print(f"Error: file not found — {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    doc_name = os.path.splitext(os.path.basename(filepath))[0]
    out_dir = os.path.join("doc_chunks", doc_name)
    os.makedirs(out_dir, exist_ok=True)

    sections = extract_sections(text)
    chunks = chunk_sections(sections)

    for chunk in chunks:
        filename = os.path.join(out_dir, f"chunk_{chunk['chunk_id'] + 1}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(chunk["text"])

    print(f"Doc     : {filepath}")
    print(f"Sections: {len(sections)}")
    print(f"Chunks  : {len(chunks)}")
    print(f"Output  : {out_dir}/")
    print()

    for chunk in chunks:
        flag = " ⚠️  oversized" if chunk["tokens"] > MAX_TOKENS else ""
        preview = chunk["text"][:60].replace("\n", " ")
        print(f"  chunk_{chunk['chunk_id'] + 1}.md  |  {chunk['tokens']} tokens  |  {preview}...{flag}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_markdown_file>")
        print("Example: python chunker.py reports/hdfc-full-report.md")
        sys.exit(1)

    chunk_file(sys.argv[1])