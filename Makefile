.PHONY: install test lint collect serve clean

install:            ## Install full local stack
	.venv/bin/pip install -r requirements.txt

test:               ## Hermetic test suite (no data/ needed)
	.venv/bin/python -m pytest tests/ -q

lint:               ## Lint everything
	.venv/bin/ruff check src api tests

collect:            ## Fetch the 80-note corpus into data/notes/
	.venv/bin/python src/collect.py

serve:              ## API on :8000
	.venv/bin/python api/main.py

clean:              ## Caches only (never data/, models/, mlruns/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
