"""Tests for the dialect-to-scientific-entity mapping layer."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.dialect_mapping.mapper import DialectMapper

LEXICON_PATH = Path(__file__).resolve().parents[1] / "data" / "dialect_lexicon.json"


@pytest.fixture(scope="module")
def mapper():
    return DialectMapper.from_lexicon_file(LEXICON_PATH)


def test_unambiguous_term_maps_to_single_entity(mapper):
    result = mapper.map_query("Mere kapas ke patton par tela lag gaya hai")
    assert result.candidate_entities == ["jassid"]
    assert not result.ambiguous


def test_normalized_query_appends_entity(mapper):
    result = mapper.map_query("safed makkhi lag gayi hai")
    assert "whitefly" in result.normalized_query
    assert result.original_query in result.normalized_query


def test_ambiguous_term_surfaces_all_candidates(mapper):
    result = mapper.map_query("Tamatar ke paudhon par jhulsa rog lag gaya hai")
    assert result.ambiguous
    assert set(result.candidate_entities) >= {"bacterial leaf blight", "late blight", "early blight"}


def test_longer_term_matched_before_substring(mapper):
    # "hara tela" should match as one unit, not "tela" + leftover "hara ".
    result = mapper.map_query("Bhindi ke patton par hara tela dikh raha hai")
    assert result.matched_terms == ["hara tela"]


def test_no_match_returns_original_query_unchanged(mapper):
    result = mapper.map_query("this query has no dialect terms in it at all")
    assert result.normalized_query == result.original_query
    assert result.candidate_entities == []
    assert not result.ambiguous


def test_case_insensitive_matching(mapper):
    result = mapper.map_query("SAFED MAKKHI lag gayi")
    assert "whitefly" in result.candidate_entities
