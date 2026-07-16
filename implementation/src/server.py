"""HTTP backend for the demo frontend -- a thin FastAPI wrapper around
AdvisoryPipeline (src/pipeline.py). Loads one pipeline instance at process
startup (so the multi-second model-load costs documented in README.md's
latency table happen once, not per request) and exposes it as /query and
/health. Does not add any retrieval/mapping/generation/gate logic of its own
-- see ../ARCHITECTURE.md for how this fits into the full frontend+backend
setup.

Run with:
    cd implementation && source .venv/bin/activate
    uvicorn src.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import AdvisoryPipeline

pipeline: AdvisoryPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = AdvisoryPipeline(use_llm=True, verbose=True)
    yield


app = FastAPI(title="Speech-Native RAG Advisory -- Demo Backend", lifespan=lifespan)

# Demo-only: allows the static frontend page (opened from disk or a separate
# dev server on another port) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if pipeline is not None else "loading",
        "llm_loaded": pipeline is not None and pipeline.llm is not None,
    }


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if pipeline is None:
        return {"error": "pipeline still loading, try again shortly"}
    result = pipeline.answer_text_query(req.text)
    return asdict(result)
