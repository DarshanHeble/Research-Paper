"""Hybrid (lexical + dense) retrieval fusion -- Section III-D / RQ4.

Fuses BM25 and dense retriever rankings with Reciprocal Rank Fusion (RRF), which is
score-scale-free (BM25 scores and cosine similarities are not directly comparable,
so fusing on rank rather than raw score avoids having to invent a normalization
scheme). RRF score for a document d given a set of rankings R:

    RRF(d) = sum_r  1 / (k + rank_r(d))

with k=60 (the standard default from the original RRF paper), applied over the
union of the top-N candidates from each individual retriever.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


@dataclass
class RetrievedPassage:
    doc_id: str
    score: float
    passage: dict
    bm25_rank: int | None = None
    dense_rank: int | None = None


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60, candidate_pool: int = 20):
        assert bm25.doc_ids == dense.doc_ids, "BM25 and dense retriever must index the same passage set/order"
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self._passages_by_id = {p["id"]: p for p in bm25.passages}

    def search(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        bm25_hits = self.bm25.search(query, top_k=self.candidate_pool)
        dense_hits = self.dense.search(query, top_k=self.candidate_pool)
        return self._fuse(bm25_hits, dense_hits, top_k=top_k)

    def search_with_query_embedding(self, query_text_for_bm25: str, query_embedding, top_k: int = 5) -> list[RetrievedPassage]:
        """Fuse BM25-over-text with dense-over-a-precomputed-embedding.

        Used by the speech-native path: there is no ASR transcript for the dense
        side (the whole point is skipping ASR), but the dialect mapper still gives
        us normalized entity strings we can lexically match with BM25.
        """
        bm25_hits = self.bm25.search(query_text_for_bm25, top_k=self.candidate_pool)
        dense_hits = self.dense.search_by_embedding(query_embedding, top_k=self.candidate_pool)
        return self._fuse(bm25_hits, dense_hits, top_k=top_k)

    def _fuse(self, bm25_hits, dense_hits, top_k: int) -> list[RetrievedPassage]:
        rrf_scores: dict[str, float] = {}
        bm25_rank_of: dict[str, int] = {}
        dense_rank_of: dict[str, int] = {}

        for rank, hit in enumerate(bm25_hits, start=1):
            rrf_scores[hit.doc_id] = rrf_scores.get(hit.doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            bm25_rank_of[hit.doc_id] = rank
        for rank, hit in enumerate(dense_hits, start=1):
            rrf_scores[hit.doc_id] = rrf_scores.get(hit.doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            dense_rank_of[hit.doc_id] = rank

        ranked_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]
        return [
            RetrievedPassage(
                doc_id=doc_id,
                score=rrf_scores[doc_id],
                passage=self._passages_by_id[doc_id],
                bm25_rank=bm25_rank_of.get(doc_id),
                dense_rank=dense_rank_of.get(doc_id),
            )
            for doc_id in ranked_ids
        ]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    kb_path = Path(__file__).resolve().parents[2] / "data" / "kb.json"
    bm25 = BM25Retriever.from_kb_file(kb_path)
    dense = DenseRetriever.from_kb_file(kb_path)
    hybrid = HybridRetriever(bm25, dense)
    q = " ".join(sys.argv[1:]) or "gulabi sundi in cotton boll"
    for r in hybrid.search(q, top_k=3):
        print(f"{r.score:6.4f}  {r.doc_id}  bm25_rank={r.bm25_rank} dense_rank={r.dense_rank}  {r.passage['title']}")
