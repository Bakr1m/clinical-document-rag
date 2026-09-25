# Model Card — Clinical Document RAG Assistant v1.0.0

## System details
- **Pipeline:** section chunking (506 chunks) → nomic-embed-text (768-dim,
  unit-norm) → FAISS exact IP → `gpt-oss:120b-cloud` (temp 0.2, 300-cap,
  verbatim refusal + exact-ID citation contract).
- **Serving:** stateless FastAPI `POST /ask` (bounds, rate limit, verified
  citations); Docker `bakr1m/rag-api:v1` (398 MB, Ollama via env).

## Intended use
- **Lookup assistance over the indexed sample corpus only.** Answers carry
  resolvable citations; refusals carry none. **Out of scope:** medical advice,
  use on real patient notes without governance, autonomous decisions.

## Data
- 80 public MTSamples reports, 10 specialties (see manifest). No PHI.

## Evaluation
- Retrieval: 8/10 manual, hit-rate@5 13/15. Correctness: 11/17 + 1 partial;
  failures named (contradiction, false refusal, wrong-note, infra flakes).
- Refusals 2/2; citations 11/13 exact post-fix.

## Limitations & ethics
- Small corpus; cloud flakiness; cosine scores uncalibrated; known
  contradiction failure exists in the logs. Production needs continuous eval,
  retries, governance (BAAs, access control, audit). See README.
