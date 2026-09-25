"""Day 48: stateless RAG API — POST /ask with validation + rate limiting.

Design (same as Project 2's scorer): the request carries everything, the
server keeps no session state. Input length is capped (cost/abuse/context
protection) and callers are rate-limited per IP (in-memory sliding window).
Generation itself lives in src/rag.ask; this layer only validates, limits,
and formats with verified citations.
"""
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cite import extract_sources, resolve_citations
from rag import ask

MAX_QUESTION_CHARS = 500
RATE_LIMIT = 20  # requests
RATE_WINDOW_S = 60.0

app = FastAPI(title="Clinical Document RAG API")

_hits: dict[str, deque] = defaultdict(deque)


def answer_question(question, k=5):
    """Single seam for tests to stub (real path calls the RAG pipeline)."""
    return ask(question, k=k)


def check_rate_limit(client_ip):
    now = time.monotonic()
    window = _hits[client_ip]
    while window and now - window[0] > RATE_WINDOW_S:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return False
    window.append(now)
    return True


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    section: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    unresolved_citations: list[str]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    try:
        res = answer_question(req.question, k=req.k)
    except Exception as e:  # noqa: BLE001 - pipeline failures -> 502, never 500
        raise HTTPException(status_code=502, detail=f"answering failed: {e}")
    if res["refused"]:
        return AskResponse(answer=res["answer"], refused=True,
                           citations=[], unresolved_citations=[])
    hits = [({k: c[k] for k in ("chunk_id", "doc_id", "section")}, c["score"])
            for c in res["contexts"]]
    verified, unresolved = resolve_citations(
        extract_sources(res["answer"]), hits)
    return AskResponse(
        answer=res["answer"],
        refused=False,
        citations=[Citation(**v) for v in verified],
        unresolved_citations=unresolved,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
