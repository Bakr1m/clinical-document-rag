# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-09-25

### Added
- Section-aware RAG over 80 clinical notes (506 chunks, FAISS exact search).
- Refusal contract + verified chunk-ID citations (prefix bug found and fixed).
- 17-QA eval set with graded correctness and named failure modes.
- Stateless FastAPI `/ask` (bounds, rate limiting, verified cites).
- CPU-lean Docker image (`bakr1m/rag-api:v1`, 398 MB), parity-verified.
- 23 hermetic tests; ruff-clean; 10 executing notebooks.
