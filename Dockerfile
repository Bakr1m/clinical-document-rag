FROM python:3.12-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + the public RAG data files only (chunks + embeddings, ~2 MB).
# Raw notes, notebooks, tests, venvs stay out (see .dockerignore).
COPY src/ ./src/
COPY api/ ./api/
COPY data/chunks.json ./data/chunks.json
COPY data/embeddings.npy ./data/embeddings.npy

EXPOSE 8000

# Ollama endpoint (embeddings + generation) comes from the environment —
# never baked into the image. No secrets live in this repo or image.
CMD ["python", "api/main.py"]
