"""RAG wiring tests: prompt contract hermetic; generation needs Ollama."""
import pytest

from src.embed import ollama_up
from src.rag import REFUSAL, SYSTEM, ask, build_prompt

HITS = [
    ({"chunk_id": "radiology_1501#EXAM#0",
      "text": "[Radiology | EXAM] No evidence of deep venous thrombosis."}, 0.77),
    ({"chunk_id": "neurology_2842#CC#5",
      "text": "[Neurology | CC] DVT one year ago, on Coumadin."}, 0.61),
]


def test_prompt_contains_contract_context_and_query():
    p = build_prompt("Was there DVT?", HITS)
    assert REFUSAL in p  # refusal instruction present
    assert "No evidence of deep venous thrombosis" in p  # context present
    assert "Was there DVT?" in p  # query present
    assert "radiology_1501#EXAM#0" in p  # chunk IDs present for citation


def test_prompt_demands_source_list():
    assert "SOURCES" in SYSTEM


@pytest.mark.skipif(not ollama_up(), reason="needs local Ollama + cloud model")
def test_live_refusal_on_unanswerable():
    res = ask("What was the patient's intergalactic travel history?", k=3)
    assert res["refused"]  # must decline, never invent
    assert REFUSAL in res["answer"]
