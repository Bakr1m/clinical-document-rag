"""Hermetic retrieval tests (synthetic vectors; live search guarded by skipif)."""
from pathlib import Path

import numpy as np
import pytest

from src.embed import ollama_up
from src.retrieve import build_index, search


def test_index_returns_nearest_first():
    rng = np.random.RandomState(0)
    mat = rng.rand(20, 16).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    index = build_index(mat)
    assert index.ntotal == 20
    q = mat[7:8].copy()
    scores, ids = index.search(q, 3)
    assert ids[0][0] == 7  # exact self-match on top
    assert scores[0][0] > scores[0][1] >= scores[0][2]


def test_ip_ranking_equals_cosine_on_unit_vectors():
    rng = np.random.RandomState(1)
    mat = rng.rand(30, 16).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    q = mat[3:4]
    index = build_index(mat)
    _, ids = index.search(q, 30)
    cosine_rank = np.argsort(-(mat @ q[0]))
    assert list(ids[0]) == list(cosine_rank)  # identical orderings


@pytest.mark.skipif(
    not ollama_up() or not Path("data/chunks.json").exists(),
    reason="needs local Ollama + data/ (gitignored; run make collect)",
)
def test_live_round_trip_finds_known_chunk():
    """A near-verbatim query must retrieve its source chunk top-1."""
    import json
    from pathlib import Path

    chunks = json.loads(Path("data/chunks.json").read_text())
    target = chunks[0]
    hits = search(target["text"][:200], k=3)
    assert hits[0][0]["chunk_id"] == target["chunk_id"]
