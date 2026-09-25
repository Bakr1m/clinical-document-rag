"""Day 45: retrieval -> generation wiring with an explicit refusal contract.

The model answers ONLY from the retrieved context and must emit
NOT_FOUND when the context doesn't contain the answer — a declined answer
beats a fabricated one, especially in a clinical setting.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve import search

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("GEN_MODEL", "gpt-oss:120b-cloud")
REFUSAL = "NOT FOUND IN THE PROVIDED NOTES"

SYSTEM = (
    "You answer questions using ONLY the clinical note excerpts below. "
    "Every factual claim must come from the excerpts. If the excerpts do not "
    f"contain the answer, reply with exactly: {REFUSAL}. "
    "End your answer with the chunk IDs you used, like [SOURCES: id1, id2]. "
    "Copy each chunk ID character-for-character exactly as shown, including "
    "the #SECTION#index suffix (e.g. radiology_1501#EXAM#0) — never shorten, "
    "reformat, or invent IDs. Be concise."
)


def build_prompt(query, hits):
    """Render context + query + refusal contract (pure, fully testable)."""
    ctx = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c, _ in hits)
    return (
        f"{SYSTEM}\n\n--- EXCERPTS ---\n{ctx}\n\n--- QUESTION ---\n{query}\n"
    )


def generate(prompt, model=MODEL, url=OLLAMA_URL, timeout=300):
    """Single Ollama generation call (low temperature, bounded length)."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False,
         "options": {"temperature": 0.2, "num_predict": 300}}
    ).encode()
    req = urllib.request.Request(
        url + "/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["response"].strip()


def ask(query, k=5):
    """Retrieve -> prompt -> generate. Returns answer + cited chunk IDs."""
    hits = search(query, k=k)
    answer = generate(build_prompt(query, hits))
    return {
        "query": query,
        "answer": answer,
        "cited_chunks": [c["chunk_id"] for c, _ in hits],
        "contexts": [
            {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
             "section": c["section"], "score": round(float(s), 4)}
            for c, s in hits
        ],
        "refused": REFUSAL in answer,
    }


def main():
    good = ask("What did the duplex venous ultrasound of the left lower "
               "extremity show regarding deep venous thrombosis?")
    print("ANSWERABLE:\n", good["answer"][:600])
    print("cited:", good["cited_chunks"][:3], "| refused:", good["refused"])
    bad = ask("What was the patient's intergalactic travel history?")
    print("\nUNANSWERABLE:\n", bad["answer"][:300])
    print("refused:", bad["refused"])
    out = PROJECT_ROOT / "models" / "rag_demo.json"
    out.write_text(json.dumps({"answerable": good, "unanswerable": bad}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
