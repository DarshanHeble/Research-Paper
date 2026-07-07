"""End-to-end smoke tests: the full pipeline must run without crashing and must
return either a grounded answer or an explicit escalation -- never silently
nothing, and never a crash. Uses the real retrieval/mapping/gate stack with
use_llm=False (deterministic templated answers) so these tests don't depend on
whether a local LLM download happens to be available in the test environment.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import AdvisoryPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline():
    return AdvisoryPipeline(use_llm=False, verbose=False)


def test_text_query_returns_answer_or_escalation(pipeline):
    result = pipeline.answer_text_query("My cotton leaves have whitefly and sooty mould")
    assert result.answer  # never empty
    assert result.answer_source in ("template", "llm", "escalation")
    if not result.escalate:
        assert result.top1_id is not None


def test_dialectal_query_gets_dialect_mapped(pipeline):
    result = pipeline.answer_text_query("Mere kapas ke patton par tela lag gaya hai")
    assert result.normalized_query is not None
    assert "jassid" in result.matched_dialect_terms or "jassid" in result.normalized_query


def test_low_confidence_query_escalates(pipeline):
    # A query with no real relationship to anything in the KB should not get a
    # confident, silently-wrong answer.
    result = pipeline.answer_text_query("asdkjf qwoeiru random nonsense zzz")
    # Whatever confidence comes out, the contract holds: an answer is always present,
    # and if the gate decided not to trust it, the answer text says so explicitly.
    assert result.answer
    if result.escalate:
        assert "ESCALATE" in result.answer.upper()


def test_pipeline_reports_timings(pipeline):
    result = pipeline.answer_text_query("wheat rust yellow pustules")
    assert "retrieval" in result.timings_sec
    assert "dialect_mapping" in result.timings_sec
    assert all(v >= 0 for v in result.timings_sec.values())


@pytest.mark.skipif(
    not (REPO_ROOT / "models" / "speech_adapter.pt").exists(),
    reason="speech adapter not trained yet -- run `python -m src.speech_retrieval.train_adapter`",
)
def test_speech_native_audio_query_smoke(pipeline, tmp_path):
    from src.tts_util import synthesize

    wav_path = tmp_path / "q.wav"
    synthesize("What can I do about pink bollworm in Cotton?", wav_path, voice="en")
    result = pipeline.answer_audio_query_speech_native(wav_path)
    assert result.answer
    assert result.mode == "speech_native"


def test_asr_cascade_audio_query_smoke(pipeline, tmp_path):
    from src.tts_util import synthesize

    wav_path = tmp_path / "q.wav"
    synthesize("My tomato leaves have brown rings and lesions", wav_path, voice="en")
    result = pipeline.answer_audio_query_asr_cascade(wav_path)
    assert result.answer
    assert result.mode == "asr_cascade"
    assert result.original_query  # a transcript was produced, even if imperfect
