# Project 4: Clinical Document RAG Assistant

**Days 41–50 | Healthcare ML Portfolio**

## Business Context

Clinicians and analysts drown in unstructured notes (discharge summaries,
radiology, operative reports). This assistant answers natural-language
questions over 80 sample clinical documents with cited, refusal-capable
responses — no manual searching, no trusted-blindly LLM output.

## Dataset

- **Source:** `harishnair04/mtsamples` (Hugging Face mirror of the public
  MTSamples sample-transcription corpus; `mtsamples.com` blocks automated
  fetching — documented fallback). 80 notes, 10 specialties, seeded sample.
- **Privacy posture:** public educational samples only — no PHI, no real
  patients, no credentials. Production on real notes would additionally need
  BAAs, access controls, encryption, audit logging, data-use approvals.
- Reproduce: `.venv/bin/python src/collect.py` → `data/notes/` + manifest.

## Approach

1. **Collection + privacy** (Day 41): stratified 10×8 corpus, manifest, note.
2. **Section chunking** (Day 42): split on `HEADER:` markers, sentence-pack
   (≤600 chars, 1-sentence overlap), `[specialty | section]`-prefixed metadata.
3. **Local embeddings** (Day 42): nomic-embed-text via Ollama, L2-normalized
   (506 × 768) so inner product == cosine.
4. **FAISS retrieval** (Day 43): exact IndexFlatIP, top-k search.
5. **Retrieval eval first** (Day 44): 10 queries manually graded — 8/10,
   misses diagnosed before any generation existed.
6. **Generation + refusal** (Day 45): `gpt-oss:120b-cloud`, temp 0.2, 300-cap,
   verbatim `NOT FOUND IN THE PROVIDED NOTES` contract.
7. **RAG eval set** (Day 46): 17 QA (traps + unanswerables) — hit-rate 13/15,
   correctness 11/17 with named failure modes.
8. **Verified citations** (Day 47): exact chunk-ID resolution; prefixes and
   refusal-citations rejected; prompt fix measured prefix → exact.
9. **API** (Day 48): stateless `POST /ask` (bounds, rate limit, verified cites).
10. **Container** (Day 49): 398 MB, env-driven Ollama endpoint, parity-verified.

## Results

| Stage | Metric |
|---|---|
| Retrieval (10 queries, top-3) | 8/10 HIT |
| Retrieval hit-rate@5 (eval set) | 13/15 |
| Answer correctness (17 QA) | 11 correct, 1 partial, 3 wrong, 2 infra-flakes |
| Refusals | 2/2 correct declines |
| Citations | 11/13 exact-verified post-fix |
| Live `/ask` | correct answer + exact citation, local == container |

## Example Q&A

**Q:** *What did the duplex venous ultrasound show regarding deep venous thrombosis?*
**A:** *No evidence of deep venous thrombosis* `[radiology_1501#EXAM#0]`

**Q:** *What chemotherapy dosage was prescribed?*
**A:** *NOT FOUND IN THE PROVIDED NOTES.* (correct refusal, cites nothing)

## Limitations

1. **Small, narrow corpus** (80 notes, 10 specialties): real coverage needs
   orders of magnitude more documents + continuous eval.
2. **Known failure modes, measured**: confident contradiction of context (q02),
   false refusal (q04), right-context-wrong-note (q14) — auditable in
   `models/eval_results.json`, not hidden.
3. **Cloud dependence + flakiness**: free-tier calls truncate/empty
   intermittently; production needs retries, timeouts, and a local fallback.
4. **No calibration of trust**: scores are cosine similarities, not
   probabilities; thresholds are heuristic.

## Ethical Considerations

- **Verify, don't trust**: every claim carries a resolvable citation; the
  README leads with the failure taxonomy, not the accuracy table.
- **Refusal as safety**: declining is engineered, tested, and measured.
- **No PHI anywhere** in this repo; production deployment has a defined
  governance checklist (see Dataset).

## Project Structure

```
project4_rag/
├ data/notes/            # 80 txt + manifest (gitignored; make collect)
├ data/chunks.json + embeddings.npy + faiss.index (gitignored; built)
├ notebooks/             # 01_collect ... 10_wrapup (all execute clean)
├ src/                   # collect, chunk, embed, retrieve, rag, cite,
│                        #   eval_retrieval, eval_rag, serve
├ eval/questions.json    # 17 QA with expected answers
├ api/main.py            # thin entrypoint (PORT-aware)
├ tests/                 # 23 hermetic tests (synthetic + guarded live)
├ models/                # *_metrics/eval JSONs (gitignored)
├ mlruns/                # (Day 46+ tracking if enabled; gitignored)
├ example via live curl (see Run with Docker)
├ Dockerfile (398 MB) + requirements-serve.txt (no torch/GPU weight)
├ requirements.txt (full) / requirements-train.txt (indexing stack)
└ README.md
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text            # local embeddings
make collect                            # 80 notes
.venv/bin/python src/chunk.py && .venv/bin/python src/embed.py
make test                               # hermetic suite
python api/main.py                      # :8000 (needs Ollama for /ask)
```

## Try It in 60 Seconds (no local setup needed)

```bash
docker pull bakr1m/rag-api:latest
docker run -d --name rag -p 8004:8000 bakr1m/rag-api:latest
curl http://localhost:8004/health
# {"status":"healthy"}
docker stop rag && docker rm rag
```

Health proves the serving layer (retrieval index + API). The `/ask`
endpoint needs a local LLM — no recruiter has one preinstalled, so:

```bash
# 1. Install + start Ollama (https://ollama.com), then:
ollama pull llama3.1
# 2. Run the API pointing at it:
docker run -d --name rag -p 8004:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434 bakr1m/rag-api:latest
curl -X POST http://localhost:8004/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What did the duplex venous ultrasound show?"}'
# -> answer + exact [source#section#chunk] citations, or a refusal
```

(`host.docker.internal` reaches your machine's Ollama from inside the
container on Docker Desktop; on native Linux use `--network host` with
`OLLAMA_URL=http://localhost:11434`.)

## Run with Docker

```bash
docker pull bakr1m/rag-api:latest
docker run -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 bakr1m/rag-api:latest
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" -d '{"question":"...?"}'
```

## Problems Encountered (Build & Deploy)

1. **No Ollama daemon in CI.** `/ask` needs a live LLM, which a GitHub
   runner doesn't have — so CI smoke-tests health only, and `/ask` is
   covered by unit tests with stubbed generation. The contract (refusal +
   citations) is tested; the prose quality is evaluated offline on 17
   graded Q&As.
2. **Citations were unverified.** The first version cited chunk IDs the
   answer didn't actually come from. Fixed with exact-verification:
   11/13 citations now match the retrieved text verbatim, and anything
   unverifiable becomes a refusal instead of a guess.
3. **Refusal as a feature.** "NOT FOUND IN THE PROVIDED NOTES" is a
   first-class correct output (2/2 correct declines in eval) — a RAG
   system that can't say "I don't know" will hallucinate citations.
4. **Two-file artifact.** Unlike the single-model repos, this one ships
   chunks + embeddings as two release assets with two SHA checks — the
   Dockerfile fails closed if either hash mismatches.

## Key Learnings

1. **Debug retrieval before generating** — else failures surface as fluent lies.
2. **Chunk boundaries are retrieval quality** (sections, never mid-fact).
3. **Refusal is a feature** with a test, not a fallback apology.
4. **Citations must resolve exactly** — prefixes are false authority.
5. **Eval sets beat vibes** — 11/17 with names beats "feels right."
6. **Stateless serving + env config** scales and keeps secrets out of code.
