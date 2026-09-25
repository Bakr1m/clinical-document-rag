"""API contract tests: validation, citations, rate limit (stubbed answers).

answer_question is stubbed so no test needs Ollama, models, or data —
one live test (skipif) covers the real pipeline end to end.
"""
import pytest
from fastapi.testclient import TestClient

from src import serve
from src.embed import ollama_up
from src.serve import app

CTX = [{"chunk_id": "radiology_1501#EXAM#0", "doc_id": "radiology_1501",
        "section": "EXAM", "score": 0.77}]


@pytest.fixture(autouse=True)
def stub_answers(monkeypatch):
    def fake_ask(question, k=5):
        return {
            "query": question,
            "answer": ("No evidence of DVT. [SOURCES: radiology_1501#EXAM#0]"),
            "cited_chunks": ["radiology_1501#EXAM#0"],
            "contexts": CTX,
            "refused": False,
        }

    monkeypatch.setattr(serve, "answer_question", fake_ask)
    serve._hits.clear()
    yield
    serve._hits.clear()


client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "healthy"}


def test_ask_returns_verified_citations():
    r = client.post("/ask", json={"question": "DVT findings?", "k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["refused"] is False
    assert body["citations"] == [
        {"chunk_id": "radiology_1501#EXAM#0", "doc_id": "radiology_1501",
         "section": "EXAM"}
    ]
    assert body["unresolved_citations"] == []


def test_ask_rejects_bad_input():
    assert client.post("/ask", json={"question": "", "k": 5}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/ask", json={"question": "ok", "k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "ok", "k": 11}).status_code == 422


def test_rate_limit_trips():
    statuses = {
        client.post("/ask", json={"question": f"q{i}"}).status_code
        for i in range(serve.RATE_LIMIT + 1)
    }
    assert statuses == {200, 429}  # first 20 pass, 21st is limited
    assert client.post("/ask", json={"question": "one more"}).status_code == 429


@pytest.mark.skipif(not ollama_up(), reason="needs local Ollama + cloud model")
def test_live_ask_end_to_end(monkeypatch):
    from src import rag

    monkeypatch.setattr(serve, "answer_question", rag.ask)
    r = client.post("/ask", json={"question": "What did the duplex venous "
                   "ultrasound show regarding deep venous thrombosis?"})
    assert r.status_code == 200, r.text
    assert 0 <= len(r.json()["citations"]) <= 5
