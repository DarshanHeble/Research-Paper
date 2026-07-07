"""Confidence-gated escalation -- Section III-E / RQ-adjacent (Component 3).

Combines a closed-book signal and an open-book signal into a single confidence
score, following the closed-book/open-book taxonomy from Dependable RAG
(dependablerag2026, see agents/components/confidence-gated-escalation.md):

  * Closed-book signal: retrieval score margin -- the normalized gap between the
    top-1 and top-2 hybrid retrieval scores. A large margin means the retriever is
    "sure" one passage is much more relevant than the runner-up; a small margin
    means the query is genuinely ambiguous between two KB entries (e.g. an
    unresolved dialectal ambiguity like "jhulsa rog").
  * Open-book signal: lexical grounding overlap -- the fraction of content words
    in the generated/templated answer that also appear in the retrieved passage
    it was generated from. Low overlap suggests the answer isn't actually
    grounded in what was retrieved (a proxy for the "compare model behavior with
    vs without context" idea in Dependable RAG, simplified to something that
    doesn't require running the generator twice).

Per INTRYGUE (intrygue2026), naive single-signal confidence scores misfire in
both directions -- this is exactly why gate.py is validated empirically against
labeled data in validate_gate.py rather than trusted by construction. Nothing in
this file should be read as "solved"; see validate_gate.py's output for whether
it actually beats an always-answer baseline on data/confidence_labels.jsonl.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "with", "this", "that", "it", "as", "at", "be", "by", "from",
    "your", "my", "you", "i", "what", "how", "do", "does", "can", "should",
    "which", "when", "if", "not", "no", "yes", "please", "me",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


@dataclass
class ConfidenceResult:
    retrieval_margin: float       # closed-book signal, in [0, 1]-ish (unbounded above 0)
    grounding_overlap: float      # open-book signal, in [0, 1]
    confidence: float             # combined score, in [0, 1]
    escalate: bool
    threshold: float


class ConfidenceGate:
    """Combines retrieval-margin (closed-book) + grounding-overlap (open-book).

    combined confidence = w_margin * normalized_margin + w_overlap * grounding_overlap

    normalized_margin squashes the raw RRF top1-top2 gap (which is small and
    unbounded, since RRF scores are sums of 1/(k+rank) terms) into [0, 1] via
    margin / (margin + saturation_point), a simple saturating transform so a
    tiny absolute gap doesn't need a hand-tuned linear scale.
    """

    def __init__(self, threshold: float = 0.5, w_margin: float = 0.5, w_overlap: float = 0.5,
                 margin_saturation: float = 0.01):
        assert abs(w_margin + w_overlap - 1.0) < 1e-6, "weights must sum to 1"
        self.threshold = threshold
        self.w_margin = w_margin
        self.w_overlap = w_overlap
        self.margin_saturation = margin_saturation

    def score_margin(self, top1_score: float, top2_score: float | None) -> float:
        if top2_score is None:
            return 1.0  # only one candidate at all -- treat as maximally confident on retrieval side
        raw_margin = max(0.0, top1_score - top2_score)
        return raw_margin / (raw_margin + self.margin_saturation)

    def score_grounding(self, answer_text: str, passage_text: str) -> float:
        answer_words = _content_words(answer_text)
        passage_words = _content_words(passage_text)
        if not answer_words:
            return 0.0
        overlap = answer_words & passage_words
        return len(overlap) / len(answer_words)

    def score(self, top1_score: float, top2_score: float | None, answer_text: str, passage_text: str) -> ConfidenceResult:
        margin = self.score_margin(top1_score, top2_score)
        overlap = self.score_grounding(answer_text, passage_text)
        combined = self.w_margin * margin + self.w_overlap * overlap
        return ConfidenceResult(
            retrieval_margin=margin,
            grounding_overlap=overlap,
            confidence=combined,
            escalate=combined < self.threshold,
            threshold=self.threshold,
        )


if __name__ == "__main__":
    gate = ConfidenceGate(threshold=0.5)
    r = gate.score(
        top1_score=0.033, top2_score=0.031,
        answer_text="This looks like pink bollworm damage to cotton bolls.",
        passage_text="Pink bollworm (Pectinophora gossypiella) larvae bore into cotton flower buds and bolls...",
    )
    print(r)
