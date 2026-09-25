"""Day 42: local embeddings via Ollama (nomic-embed-text).

Zero extra dependencies (urllib only). Vectors are L2-normalized so FAISS
inner-product ranking equals cosine similarity (Day 43).

Output: data/embeddings.npy (float32, rows aligned with data/chunks.json)
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Endpoint of the Ollama daemon (embeddings + generation). Env-overridable so
# containers never bake in hostnames, ports, or credentials (Day 49).
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
OUT_PATH = PROJECT_ROOT / "data" / "embeddings.npy"
BATCH = 32


def ollama_up(url=OLLAMA_URL, timeout=5):
    try:
        urllib.request.urlopen(url + "/api/tags", timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return False


def embed_texts(texts, model=MODEL, url=OLLAMA_URL):
    """Batch embed via /api/embed; returns L2-normalized float32 matrix."""
    vecs = []
    for i in range(0, len(texts), BATCH):
        payload = json.dumps(
            {"model": model, "input": texts[i : i + BATCH]}
        ).encode()
        req = urllib.request.Request(
            url + "/api/embed", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            vecs.extend(json.load(r)["embeddings"])
    mat = np.asarray(vecs, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat


def main():
    assert ollama_up(), "Ollama not reachable at localhost:11434"
    chunks = json.loads(CHUNKS_PATH.read_text())
    mat = embed_texts([c["text"] for c in chunks])
    np.save(OUT_PATH, mat)
    print(f"chunks: {len(chunks)} -> embeddings {mat.shape} (dim {mat.shape[1]})")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
