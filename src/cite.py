"""Day 47: verified source citations.

Parses the model's [SOURCES: ...] list and resolves every ID against the
retrieved chunks. Exact chunk-ID match = verified; prefix/unknown IDs are
reported as UNRESOLVED (never silently accepted — an unverifiable citation
is worse than none, it lends false authority).
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SOURCES_RE = re.compile(r"\[SOURCES:\s*([^\]]+)\]", re.IGNORECASE)


def extract_sources(answer):
    """Chunk IDs the model claimed, in order (empty list if none)."""
    m = SOURCES_RE.search(answer or "")
    if not m:
        return []
    return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]


def resolve_citations(claimed, hits):
    """Split claimed IDs into verified (exact chunk match) vs unresolved."""
    valid = {c["chunk_id"]: c for c, _ in hits}
    verified, unresolved = [], []
    for cid in claimed:
        if cid in valid:
            c = valid[cid]
            verified.append({"chunk_id": cid, "doc_id": c["doc_id"],
                             "section": c["section"]})
        else:
            unresolved.append(cid)
    return verified, unresolved


def check_eval_results(path=PROJECT_ROOT / "models" / "eval_results.json"):
    """Audit every graded answer's citations. Returns the report dict."""
    report = []
    for row in json.loads(Path(path).read_text()):
        if not row["answer"].strip():
            report.append({"id": row["id"], "claimed": [], "verified": [],
                           "unresolved": [], "note": "empty answer"})
            continue
        claimed = extract_sources(row["answer"])
        # Re-resolve against the chunks cited at generation time.
        hits = [( {"chunk_id": c, "doc_id": c.split("#")[0],
                   "section": c.split("#")[1] if "#" in c else "?"},
                  0.0) for c in row["cited_chunks"]]
        verified, unresolved = resolve_citations(claimed, hits)
        report.append({"id": row["id"], "claimed": claimed,
                       "verified": [v["chunk_id"] for v in verified],
                       "unresolved": unresolved})
    return report


def main():
    report = check_eval_results()
    n_claimed = sum(len(r["claimed"]) for r in report)
    n_unres = sum(len(r["unresolved"]) for r in report)
    print(f"cited IDs claimed: {n_claimed}, unresolved: {n_unres}")
    for r in report:
        if r["unresolved"]:
            print(f"  {r['id']}: unresolved {r['unresolved']}")
    out = PROJECT_ROOT / "models" / "citation_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
