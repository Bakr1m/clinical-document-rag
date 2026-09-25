# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
make collect && make test
```

1. **No large/generated files in git**: `data/`, `models/`, `mlruns/`
   stay ignored. Only code, docs, eval questions, and notebooks are versioned.
2. **Tests are hermetic**: synthetic fixtures or Ollama-guarded live tests —
   the suite must pass on a clean checkout with no data and no daemon.
3. **Lint clean** (`ruff check src api tests`) before every push.
4. **Eval-gated changes**: chunking/retrieval/prompt edits must re-run the
   eval set and report hit-rate + correctness deltas, not vibes.
