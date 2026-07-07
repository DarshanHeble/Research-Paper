"""Tests for the confidence gate's scoring mechanics (not its real-world validity --
see src/confidence/validate_gate.py and the README for that empirical check)."""
from __future__ import annotations

from src.confidence.gate import ConfidenceGate


def test_large_margin_and_high_overlap_yields_high_confidence():
    gate = ConfidenceGate(threshold=0.5)
    result = gate.score(
        top1_score=0.05, top2_score=0.001,
        answer_text="pink bollworm larvae bore into cotton bolls",
        passage_text="pink bollworm larvae bore into cotton flower buds and bolls",
    )
    assert result.confidence > 0.5
    assert not result.escalate


def test_small_margin_and_low_overlap_yields_low_confidence_and_escalates():
    gate = ConfidenceGate(threshold=0.5)
    result = gate.score(
        top1_score=0.0301, top2_score=0.0300,
        answer_text="completely unrelated words about nothing relevant here",
        passage_text="pink bollworm larvae bore into cotton flower buds and bolls",
    )
    assert result.confidence < 0.5
    assert result.escalate


def test_single_candidate_no_runner_up_gets_max_margin():
    gate = ConfidenceGate()
    assert gate.score_margin(top1_score=0.5, top2_score=None) == 1.0


def test_grounding_overlap_is_one_for_verbatim_quote():
    gate = ConfidenceGate()
    passage = "yellow stem borer causes deadheart and whitehead symptoms in rice"
    overlap = gate.score_grounding(answer_text=passage, passage_text=passage)
    assert overlap == 1.0


def test_grounding_overlap_is_zero_for_disjoint_text():
    gate = ConfidenceGate()
    overlap = gate.score_grounding(
        answer_text="zebra giraffe elephant lion",
        passage_text="rice wheat cotton tomato",
    )
    assert overlap == 0.0


def test_threshold_is_configurable():
    gate_strict = ConfidenceGate(threshold=0.9)
    gate_lenient = ConfidenceGate(threshold=0.1)
    result_strict = gate_strict.score(0.02, 0.019, "some words in common", "some words in common here")
    result_lenient = gate_lenient.score(0.02, 0.019, "some words in common", "some words in common here")
    assert result_strict.escalate
    assert not result_lenient.escalate
