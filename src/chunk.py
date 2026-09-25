"""Day 42: section-aware chunking (never cut a clinical fact in half).

Splits each note on its `HEADER:` section markers (FINDINGS, IMPRESSION, ...),
then packs sentences into <= MAX_CHARS chunks with 1-sentence overlap.
Every chunk carries (doc_id, specialty, section) metadata for citations.

Output: data/chunks.json
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NOTES_DIR = PROJECT_ROOT / "data" / "notes"
OUT_PATH = PROJECT_ROOT / "data" / "chunks.json"

MAX_CHARS = 600
OVERLAP_SENTENCES = 1
HEADER_RE = re.compile(r"^([A-Z][A-Z /&\-\(\)]+):\s*(.*)$")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def split_sentences(text):
    return [s.strip() for s in SENT_RE.split(text.strip()) if s.strip()]


def pack_sentences(sentences, max_chars=MAX_CHARS, overlap=OVERLAP_SENTENCES):
    """Greedy sentence packing with overlap; a single long sentence stands alone."""
    chunks, cur = [], []
    cur_len = 0
    for s in sentences:
        if cur and cur_len + 1 + len(s) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def split_sections(body):
    """Yield (section, text); lines before the first header go to 'HEADER'."""
    sections, current, buf = [], "HEADER", []
    for line in body.splitlines():
        m = HEADER_RE.match(line.strip())
        if m:
            if buf:
                sections.append((current, " ".join(buf)))
            current, rest = m.group(1).strip(), m.group(2).strip()
            buf = [rest] if rest else []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((current, " ".join(buf)))
    return sections


def chunk_note(doc_id, specialty, text):
    """Full note -> list of chunk dicts (section-aware, sentence-packed)."""
    _header, _, body = text.partition("\n\n")
    chunks = []
    for section, sec_text in split_sections(body):
        for i, piece in enumerate(pack_sentences(split_sentences(sec_text))):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}#{section}#{i}",
                    "doc_id": doc_id,
                    "specialty": specialty,
                    "section": section,
                    "text": f"[{specialty} | {section}] {piece}",
                }
            )
    return chunks


def main():
    files = sorted(NOTES_DIR.rglob("*.txt"))
    assert files, f"no notes in {NOTES_DIR} — run make collect"
    all_chunks = []
    for fp in files:
        if fp.name == "manifest.json":
            continue
        text = fp.read_text()
        header, _, _ = text.partition("\n\n")
        spec = next(
            (ln.split(":", 1)[1].strip() for ln in header.splitlines()
             if ln.startswith("SPECIALTY:")), fp.parent.name,
        )
        all_chunks.extend(chunk_note(fp.stem, spec, text))
    OUT_PATH.write_text(json.dumps(all_chunks, indent=2))
    lens = [len(c["text"]) for c in all_chunks]
    print(f"notes: {len(files)} -> chunks: {len(all_chunks)} "
          f"(avg {sum(lens)/len(lens):.0f} chars, max {max(lens)})")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
