"""Hermetic collect tests (pure helpers + manifest consistency when present)."""
from pathlib import Path

import pytest

from src.collect import SPECIALTIES, manifest_consistency, slug

NOTES_DIR = Path("data/notes")


def test_slug_normalizes_specialty():
    assert slug(" Cardiovascular / Pulmonary") == "cardiovascular_pulmonary"
    assert slug("SOAP / Chart / Progress Notes") == "soap_chart_progress_notes"


def test_ten_specialties_eight_notes_each():
    assert len(SPECIALTIES) == 10


@pytest.mark.skipif(
    not (NOTES_DIR / "manifest.json").exists(),
    reason="needs data/notes (gitignored; run make collect)",
)
def test_manifest_matches_files_on_disk():
    assert manifest_consistency(NOTES_DIR)
