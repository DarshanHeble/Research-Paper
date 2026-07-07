"""Dense (embedding) retrieval over the agricultural knowledge base.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) -- small enough to run
comfortably on the 6GB RTX 3050 target hardware (Section III-F) alongside the other
pipeline stages, per the offline-deployment component brief.

Note: this module and speech_retrieval/adapter.py both target this SAME 384-dim
embedding space -- the adapter is trained to project speech encoder output into it
(Section III-B), so this retriever's `.encode()` output is the "frozen text
retriever" embedding space referenced in agents/components/speech-native-retrieval.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@dataclass
class RetrievedPassage:
    doc_id: str
    score: float
    passage: dict


class DenseRetriever:
    def __init__(self, passages: list[dict], model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.passages = passages
        self.doc_ids = [p["id"] for p in passages]
        self.model_name = model_name
        self.device = device or _pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)

        corpus_texts = [
            " ".join([p.get("title", ""), p.get("text", "")]) for p in passages
        ]
        self.doc_embeddings = self.model.encode(
            corpus_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    @classmethod
    def from_kb_file(cls, kb_path: str | Path, **kwargs) -> "DenseRetriever":
        with open(kb_path, encoding="utf-8") as f:
            passages = json.load(f)
        return cls(passages, **kwargs)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]

    def search(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        q_emb = self.encode_query(query)
        return self.search_by_embedding(q_emb, top_k=top_k)

    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedPassage]:
        """Cosine similarity search directly from a query embedding.

        Used both by text queries (via encode_query) and by the speech-native
        retriever, which produces an embedding in this same space via the trained
        adapter without ever calling encode_query/ASR.
        """
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores = self.doc_embeddings @ q
        ranked = np.argsort(-scores)[:top_k]
        return [
            RetrievedPassage(doc_id=self.doc_ids[i], score=float(scores[i]), passage=self.passages[i])
            for i in ranked
        ]


def _pick_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    import sys

    kb_path = Path(__file__).resolve().parents[2] / "data" / "kb.json"
    retriever = DenseRetriever.from_kb_file(kb_path)
    q = " ".join(sys.argv[1:]) or "small green bugs sucking sap on wheat ears"
    for r in retriever.search(q, top_k=3):
        print(f"{r.score:6.3f}  {r.doc_id}  {r.passage['title']}")
