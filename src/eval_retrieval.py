"""Day 44: manual retrieval evaluation over 10 grounded queries.

Runs each query (top-3) and saves raw results; grading is manual (read the
chunks!) and recorded in models/retrieval_eval.json by the evaluator.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve import search

OUT_PATH = PROJECT_ROOT / "models" / "retrieval_raw.json"

QUERIES = [
    "deep venous thrombosis ultrasound findings",
    "carotid artery stenosis ultrasound evaluation",
    "chest X-ray after ASD surgery atelectasis",
    "exercise induced ischemia high myocardial workload",
    "mitral regurgitation left ventricular hypertrophy echocardiogram",
    "degenerative disc disease foraminal compromise spine",
    "COPD bronchospasm three day history emergency",
    "BRCA-2 mutation breast cancer family history",
    "plantar fasciitis heel spur injection treatment",
    "migraine headache neurological assessment",
]


def main():
    results = []
    for q in QUERIES:
        hits = search(q, k=3)
        results.append(
            {
                "query": q,
                "hits": [
                    {"chunk_id": c["chunk_id"], "score": round(s, 3),
                     "snippet": c["text"][:160]}
                    for c, s in hits
                ],
            }
        )
        print(f"Q: {q}")
        for h in results[-1]["hits"]:
            print(f"  [{h['score']}] {h['chunk_id']}: {h['snippet'][:100]}...")
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
