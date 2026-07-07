"""Lexical retrieval over the agricultural knowledge base using BM25.

This is the "keyword" baseline referenced throughout Section III-D / IV-C of the
paper: it wins on rare, exact domain-critical entity names (pesticide/pest/disease
names) because BM25 rewards exact token overlap, but it will not generalize across
paraphrase or synonymy the way the dense retriever does.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple, deterministic lowercase alnum tokenizer.

    Deliberately dumb (no stemming/lemmatization) -- BM25's exact-match behavior is
    exactly the property we want to characterize honestly here, not paper over.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedPassage:
    doc_id: str
    score: float
    passage: dict


class BM25Retriever:
    """BM25 lexical retriever over a list of KB passage dicts (see data/kb.json)."""

    def __init__(self, passages: list[dict]):
        self.passages = passages
        self.doc_ids = [p["id"] for p in passages]
        # Index over title + text + entities so exact entity-name matches score highly.
        self._corpus_texts = [
            " ".join([p.get("title", ""), p.get("text", ""), " ".join(p.get("entities", []))])
            for p in passages
        ]
        self._tokenized_corpus = [tokenize(t) for t in self._corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    @classmethod
    def from_kb_file(cls, kb_path: str | Path) -> "BM25Retriever":
        with open(kb_path, encoding="utf-8") as f:
            passages = json.load(f)
        return cls(passages)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedPassage(doc_id=self.doc_ids[i], score=float(scores[i]), passage=self.passages[i])
            for i in ranked
        ]


if __name__ == "__main__":
    import sys

    kb_path = Path(__file__).resolve().parents[2] / "data" / "kb.json"
    retriever = BM25Retriever.from_kb_file(kb_path)
    q = " ".join(sys.argv[1:]) or "whitefly on cotton leaves"
    for r in retriever.search(q, top_k=3):
        print(f"{r.score:6.3f}  {r.doc_id}  {r.passage['title']}")
