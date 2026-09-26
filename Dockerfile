FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code only (no raw notes, notebooks/, tests/).
COPY src/ ./src/
COPY api/ ./api/

# RAG data: fetched by exact release version, SHA256-verified.
# Never a moving tag, never baked from a developer laptop. Provenance:
# https://github.com/Bakr1m/clinical-document-rag/releases/tag/v1.0.0
ARG DATA_TAG=v1.0.0
ARG CHUNKS_SHA256=fdcd2a2463e4c31cbb44b4a0dc376a2447fa7fff0d114c89127f3ccbee9c8687
ARG EMB_SHA256=5011762dd1ef57e072e539d32e2ee4404fcb5f706c6537fee70176c719775935
RUN mkdir -p data && \
    curl -fsSL -o data/chunks.json \
      "https://github.com/Bakr1m/clinical-document-rag/releases/download/${DATA_TAG}/chunks.json" && \
    curl -fsSL -o data/embeddings.npy \
      "https://github.com/Bakr1m/clinical-document-rag/releases/download/${DATA_TAG}/embeddings.npy" && \
    echo "${CHUNKS_SHA256}  data/chunks.json" | sha256sum -c - && \
    echo "${EMB_SHA256}  data/embeddings.npy" | sha256sum -c - && \
    python -c "import json,numpy as np; c=json.load(open('data/chunks.json')); m=np.load('data/embeddings.npy'); assert len(c)==len(m)==506, (len(c),m.shape); print('RAG data OK:', len(c), 'chunks')"

EXPOSE 8000

# Ollama endpoint (embeddings + generation) comes from the environment —
# never baked into the image. No secrets live in this repo or image.
CMD ["python", "api/main.py"]
