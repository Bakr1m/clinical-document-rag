"""Day 43: FAISS vector store + top-k similarity search.

IndexFlatIP over L2-normalized vectors: inner-product ranking == cosine
ranking, exact (no approximation) at our 506-vector scale.
Chunks live in data/chunks.json (index positions align with embedding rows).

Output: data/faiss.index
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embed import embed_texts

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
EMB_PATH = PROJECT_ROOT / "data" / "embeddings.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "faiss.index"


def build_index(mat):
    """Exact inner-product index (== cosine on unit vectors)."""
    mat = np.ascontiguousarray(mat.astype(np.float32))
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index


def load_store():
    chunks = json.loads(CHUNKS_PATH.read_text())
    mat = np.load(EMB_PATH)
    assert len(chunks) == len(mat), "chunks/embeddings row mismatch"
    return chunks, mat


def search(query, k=5, index=None, chunks=None, mat=None):
    """Embed the query, return top-k (chunk, cosine-score) pairs."""
    if index is None or chunks is None:
        chunks, mat = load_store()
        index = build_index(mat)
    q = embed_texts([query])
    scores, ids = index.search(q, k)
    return [(chunks[i], float(scores[0][r])) for r, i in enumerate(ids[0])]


def main():
    chunks, mat = load_store()
    index = build_index(mat)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"indexed {index.ntotal} vectors (dim {index.d}) -> {INDEX_PATH}")
    demo = search("deep venous thrombosis ultrasound findings", k=3,
                  index=index, chunks=chunks)
    for c, s in demo:
        print(f"  [{s:.3f}] {c['chunk_id']}: {c['text'][:90]}...")


if __name__ == "__main__":
    main()
