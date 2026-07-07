#!/usr/bin/env python
"""Builds data/confidence_labels.jsonl by running the actual pipeline (hybrid
retrieval + dialect mapping + templated answer) over data/eval_queries.jsonl and
RULE-LABELING correctness against the gold passage ids already curated in that
file (top1_id in gold_ids -> correct). This is the "hand-labeled/rule-labeled
correctness against kb.json ground truth" construction method named in the task
spec for data/confidence_labels.jsonl -- rule-based rather than a human expert
panel, since no such panel exists for this project; documented here rather than
silently presented as expert-reviewed.

Deliberately uses use_llm=False (deterministic templated answers) so labels are
reproducible run-to-run and don't depend on whether a local LLM happens to be
loadable in a given environment -- the confidence gate's grounding-overlap
signal only needs SOME answer text to compare against the retrieved passage, and
the templated answer is a direct quote of the retrieved passage, which is a
harder (less informative) test of the grounding signal than an LLM paraphrase
would be. That's a conservative choice, not a favorable one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import AdvisoryPipeline  # noqa: E402

EVAL_QUERIES_PATH = REPO_ROOT / "data" / "eval_queries.jsonl"
OUT_PATH = REPO_ROOT / "data" / "confidence_labels.jsonl"


def main():
    pipeline = AdvisoryPipeline(use_llm=False, confidence_threshold=0.0, verbose=True)
    # threshold=0.0 here means "never escalate" purely so we can observe the raw
    # top1/top2 scores and a real templated answer for every query, regardless of
    # what a downstream chosen threshold would decide -- validate_gate.py is where
    # threshold choices actually get evaluated.

    queries = []
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    labels = []
    for q in queries:
        result = pipeline.answer_text_query(q["text"], top_k=5)
        correct = result.top1_id in q["gold_ids"]
        labels.append({
            "query_id": q["id"],
            "query": q["text"],
            "phrasing": q["phrasing"],
            "rarity": q["rarity"],
            "gold_ids": q["gold_ids"],
            "top1_id": result.top1_id,
            "top1_score": result.top1_score,
            "top2_score": result.top2_score,
            "answer": result.answer,
            "answer_source": result.answer_source,
            "correct": correct,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row) + "\n")

    n_correct = sum(1 for l in labels if l["correct"])
    print(f"\nWrote {len(labels)} labeled examples to {OUT_PATH}")
    print(f"correct={n_correct} incorrect={len(labels) - n_correct} "
          f"(base top-1 accuracy on this set = {n_correct/len(labels):.3f})")


if __name__ == "__main__":
    main()
