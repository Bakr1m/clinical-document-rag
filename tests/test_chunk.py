"""Hermetic chunker tests (synthetic notes) + live embed test (skipped w/o Ollama)."""
import numpy as np
import pytest

from src.chunk import chunk_note, pack_sentences, split_sections
from src.embed import embed_texts, ollama_up

NOTE = """SPECIALTY: Radiology

FINDINGS: The lungs are clear. No focal consolidation is seen. The heart is
normal in size.
IMPRESSION: No acute disease. Follow up in six months for routine screening
of the stable nodule seen previously."""


def test_sections_split_without_loss():
    sections = split_sections(NOTE.split("\n\n", 1)[1])
    assert [s for s, _ in sections] == ["FINDINGS", "IMPRESSION"]
    joined = " ".join(t for _, t in sections)
    for frag in ["lungs are clear", "No acute disease", "stable nodule"]:
        assert frag in joined


def test_no_mid_sentence_cuts():
    chunks = chunk_note("d1", "Radiology", NOTE)
    assert len(chunks) >= 2  # findings + impression stay separate
    for c in chunks:
        assert c["text"].rstrip()[-1] in ".!?"  # every chunk ends cleanly
        assert c["chunk_id"].startswith("d1#")
    ids = [c["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)


def test_long_section_packs_with_overlap():
    sents = [f"Sentence number {i} with clinical content." for i in range(20)]
    pieces = pack_sentences(sents, max_chars=120, overlap=1)
    assert all(len(p) <= 120 or len(p.split(". ")) == 1 for p in pieces)
    assert pieces[0].split(". ")[-1] in pieces[1]  # overlap sentence shared


def test_chunk_text_carries_retrieval_context():
    chunks = chunk_note("d1", "Radiology", NOTE)
    assert all(c["text"].startswith("[Radiology | ") for c in chunks)


@pytest.mark.skipif(not ollama_up(), reason="needs local Ollama")
def test_embed_live_unit_norms():
    mat = embed_texts(["No evidence of deep venous thrombosis.", "hello world"])
    assert mat.shape[0] == 2 and mat.dtype == np.float32
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)
