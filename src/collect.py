"""Day 41: collect ~80 public MTSamples transcription reports.

Source: `harishnair04/mtsamples` (Hugging Face mirror of the public MTSamples
corpus; mtsamples.com blocks automated fetching with 403, so the mirror is
the documented fallback). Stratified sample: 10 specialties x 8 notes.
Writes data/notes/<specialty>/<id>.txt + data/notes/manifest.json.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NOTES_DIR = PROJECT_ROOT / "data" / "notes"
MANIFEST = NOTES_DIR / "manifest.json"

SPECIALTIES = [
    " Cardiovascular / Pulmonary",
    " Radiology",
    " General Medicine",
    " Gastroenterology",
    " Neurology",
    " Discharge Summary",
    " Consult - History and Phy.",
    " Orthopedic",
    " Urology",
    " SOAP / Chart / Progress Notes",
]
PER_SPECIALTY = 8
SEED = 42


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def manifest_consistency(notes_dir=NOTES_DIR):
    """Manifest lists exactly the files on disk (no orphans either way)."""
    manifest = json.loads((Path(notes_dir) / "manifest.json").read_text())
    listed = {d["doc_id"] for d in manifest["documents"]}
    on_disk = {p.stem for p in Path(notes_dir).rglob("*.txt")}
    return listed == on_disk and len(listed) > 0


def main():
    from datasets import load_dataset

    ds = load_dataset("harishnair04/mtsamples", split="train")
    rng_state = __import__("random").Random(SEED)
    manifest = {
        "source": "harishnair04/mtsamples (Hugging Face mirror of public MTSamples)",
        "fallback_reason": "mtsamples.com returns 403 to automated fetching",
        "seed": SEED,
        "documents": [],
    }
    total = 0
    for spec in SPECIALTIES:
        idx = [i for i, s in enumerate(ds["medical_specialty"]) if s == spec]
        assert len(idx) >= PER_SPECIALTY, f"not enough rows for {spec!r}"
        picks = rng_state.sample(idx, PER_SPECIALTY)
        spec_dir = NOTES_DIR / slug(spec)
        spec_dir.mkdir(parents=True, exist_ok=True)
        for i in picks:
            row = ds[i]
            doc_id = f"{slug(spec)}_{i}"
            body = (
                f"SPECIALTY: {row['medical_specialty'].strip()}\n"
                f"SAMPLE: {row['sample_name'].strip()}\n"
                f"DESCRIPTION: {row['description'].strip()}\n\n"
                f"{row['transcription'].strip()}\n"
            )
            (spec_dir / f"{doc_id}.txt").write_text(body)
            manifest["documents"].append(
                {"doc_id": doc_id, "specialty": spec.strip(),
                 "sample_name": row["sample_name"].strip()}
            )
            total += 1
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"collected {total} notes across {len(SPECIALTIES)} specialties -> {NOTES_DIR}")


if __name__ == "__main__":
    main()
