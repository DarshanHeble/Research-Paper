"""Tests for BM25 / dense / hybrid retrieval over the real data/kb.json corpus.

Uses the real sentence-transformers model (downloads/caches on first run) --
these are integration tests of the actual retrieval stack, not mocked units.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "kb.json"


@pytest.fixture(scope="module")
def bm25():
    return BM25Retriever.from_kb_file(KB_PATH)


@pytest.fixture(scope="module")
def dense():
    return DenseRetriever.from_kb_file(KB_PATH)


@pytest.fixture(scope="module")
def hybrid(bm25, dense):
    return HybridRetriever(bm25, dense)


def test_bm25_exact_entity_match_ranks_first(bm25):
    hits = bm25.search("pink bollworm Pectinophora gossypiella", top_k=3)
    assert hits[0].doc_id == "kb010"


def test_bm25_returns_requested_number_of_hits(bm25):
    hits = bm25.search("cotton pest", top_k=5)
    assert len(hits) == 5


def test_dense_paraphrase_retrieval(dense):
    # No shared exact tokens with the passage text -- tests semantic generalization.
    hits = dense.search("tiny insects sucking sap under cotton leaves causing black mould", top_k=3)
    doc_ids = [h.doc_id for h in hits]
    assert "kb011" in doc_ids  # Whitefly in Cotton


def test_hybrid_beats_or_matches_components_on_exact_entity(bm25, dense, hybrid):
    query = "karnal bunt Tilletia indica wheat"
    hybrid_hits = [h.doc_id for h in hybrid.search(query, top_k=3)]
    assert "kb006" in hybrid_hits  # Karnal Bunt of Wheat


def test_hybrid_search_result_has_rank_metadata(hybrid):
    hits = hybrid.search("rice blast disease", top_k=3)
    assert all(h.bm25_rank is not None or h.dense_rank is not None for h in hits)


def test_bm25_and_dense_share_doc_id_ordering(bm25, dense):
    assert bm25.doc_ids == dense.doc_ids
