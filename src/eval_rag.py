"""Day 46: scripted RAG eval — retrieval hit-rate@5 (automated) + answers
saved for manual correctness grading. Run: .venv/bin/python src/eval_rag.py
Output: models/eval_results.json (grades filled in by the evaluator).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag import ask
from retrieve import search

Q_PATH = PROJECT_ROOT / "eval" / "questions.json"
OUT_PATH = PROJECT_ROOT / "models" / "eval_results.json"
K = 5


def main():
    questions = json.loads(Q_PATH.read_text())["questions"]
    results = []
    for q in questions:
        hits = search(q["question"], k=K)
        hit_ids = [c["chunk_id"] for c, _ in hits]
        hit = q["doc"] is not None and any(
            h.startswith(q["doc"]) for h in hit_ids
        )
        print(f"{q['id']}: hit={hit} -> asking...", flush=True)
        res = ask(q["question"], k=K)
        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected": q["answer"],
                "retrieval_hit_at_5": hit,
                "answer": res["answer"],
                "cited_chunks": res["cited_chunks"],
                "refused": res["refused"],
                "grade": None,  # filled by manual grading below
            }
        )
    OUT_PATH.write_text(json.dumps(results, indent=2))
    hits = sum(r["retrieval_hit_at_5"] for r in results if r["expected"] != "REFUSE")
    print(f"retrieval hit-rate@5: {hits}/15")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
