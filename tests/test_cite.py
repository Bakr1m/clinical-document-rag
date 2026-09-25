"""Citation unit tests: extraction + resolution are pure and hermetic."""
from src.cite import extract_sources, resolve_citations

HITS = [
    ({"chunk_id": "radiology_1501#EXAM#0", "doc_id": "radiology_1501",
      "section": "EXAM"}, 0.77),
    ({"chunk_id": "neurology_2842#CC#5", "doc_id": "neurology_2842",
      "section": "CC"}, 0.61),
]


def test_extract_sources_parses_list():
    assert extract_sources("... disease. [SOURCES: a#X#0, b#Y#1]") == ["a#X#0", "b#Y#1"]


def test_extract_sources_empty_when_absent():
    assert extract_sources("Plain answer, no citation.") == []


def test_resolve_keeps_exact_drops_prefix():
    verified, unresolved = resolve_citations(
        ["radiology_1501#EXAM#0", "radiology_1501", "invented#X#9"], HITS)
    assert [v["chunk_id"] for v in verified] == ["radiology_1501#EXAM#0"]
    assert unresolved == ["radiology_1501", "invented#X#9"]
    assert verified[0]["section"] == "EXAM"  # doc+section travel with the ID


def test_resolve_empty_claims():
    assert resolve_citations([], HITS) == ([], [])
