#!/usr/bin/env python
"""Builds a confidence-labels JSONL by running the actual pipeline (hybrid
retrieval + dialect mapping + an answer) over data/eval_queries.jsonl and
RULE-LABELING correctness against the gold passage ids already curated in that
file (top1_id in gold_ids -> correct). This is the "hand-labeled/rule-labeled
correctness against kb.json ground truth" construction method named in the task
spec for data/confidence_labels.jsonl -- rule-based rather than a human expert
panel, since no such panel exists for this project; documented here rather than
silently presented as expert-reviewed. The correctness label is a property of
retrieval (did the top-1 passage match a gold id), independent of which answer
generation path is used below, so it stays comparable across the two variants.

Two variants, selected by --llm:

  * Default (use_llm=False, deterministic templated answers): reproducible
    run-to-run and doesn't depend on whether a local LLM happens to be
    loadable in a given environment. But the templated answer is a direct
    quote of the retrieved passage, which validate_gate.py's own diagnostic
    output documents as an artificially easy (uninformative) case for the
    open-book grounding_overlap signal: it is "grounded" by construction even
    when the retrieved passage is the wrong one. Written to
    data/confidence_labels.jsonl.
  * --llm (use_llm=True, real Qwen2.5-0.5B-Instruct generation, grounded-only
    prompting per src/generation/llm_generator.py): a genuine test of whether
    grounding_overlap can actually diverge from correctness once the answer is
    a real paraphrase rather than a verbatim quote -- this is the
    "re-validation of the confidence gate's open-book signal against
    non-template-quoting LLM answers" item main.tex's Conclusion names as
    future work. Written to data/confidence_labels_llm.jsonl, so the original
    templated file is never overwritten and both remain available for
    side-by-side comparison in validate_gate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import AdvisoryPipeline  # noqa: E402

EVAL_QUERIES_PATH = REPO_ROOT / "data" / "eval_queries.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true",
                         help="Use real LLM generation instead of the templated "
                              "verbatim-quote answer; writes to "
                              "data/confidence_labels_llm.jsonl instead of "
                              "data/confidence_labels.jsonl.")
    args = parser.parse_args()

    out_path = REPO_ROOT / "data" / ("confidence_labels_llm.jsonl" if args.llm else "confidence_labels.jsonl")

    pipeline = AdvisoryPipeline(use_llm=args.llm, confidence_threshold=0.0, verbose=True)
    # threshold=0.0 here means "never escalate" purely so we can observe the raw
    # top1/top2 scores and a real answer for every query, regardless of what a
    # downstream chosen threshold would decide -- validate_gate.py is where
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

    with open(out_path, "w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row) + "\n")

    n_correct = sum(1 for l in labels if l["correct"])
    n_llm = sum(1 for l in labels if l["answer_source"] == "llm")
    print(f"\nWrote {len(labels)} labeled examples to {out_path}")
    print(f"correct={n_correct} incorrect={len(labels) - n_correct} "
          f"(base top-1 accuracy on this set = {n_correct/len(labels):.3f})")
    if args.llm:
        print(f"answer_source=llm for {n_llm}/{len(labels)} examples "
              f"({len(labels) - n_llm} fell back to template, e.g. if the LLM failed at inference time)")


if __name__ == "__main__":
    main()
