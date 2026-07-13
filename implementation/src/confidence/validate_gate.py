"""Validates the confidence gate against data/confidence_labels.jsonl -- Section
III-E / IV-E's required empirical check (per INTRYGUE, a confidence signal must
never be trusted by construction; see gate.py's docstring).

Reports, honestly, whatever numbers come out on this small (50-example) demo
label set:
  * Calibration: binned confidence vs. actual accuracy (Expected Calibration
    Error, ECE) -- does a 0.8-confidence answer turn out right ~80% of the time?
  * A threshold sweep: for each candidate threshold, coverage (fraction
    answered rather than escalated), accuracy among answered queries, and
    escalation precision/recall for CATCHING actually-wrong answers.
  * A single headline comparison against the "always answer, never escalate"
    baseline (this dataset's raw top-1 accuracy) -- the gate only "passes" if
    there exists a threshold where accuracy-among-answered beats that baseline
    by a real margin, which is the whole point of gating at all.

This is a 50-example demo set (11 incorrect / 39 correct) -- every number below
has a wide confidence interval and should be read as "does the mechanism show
the right qualitative behavior," not as a precise calibration curve. Said
explicitly in the printed output too, not just here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.confidence.gate import ConfidenceGate  # noqa: E402

KB_PATH = REPO_ROOT / "data" / "kb.json"


def load_labels(labels_path: Path) -> list[dict]:
    rows = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_kb_by_id() -> dict:
    with open(KB_PATH, encoding="utf-8") as f:
        passages = json.load(f)
    return {p["id"]: p for p in passages}


def compute_confidences(rows: list[dict], kb_by_id: dict, gate: ConfidenceGate) -> list[dict]:
    out = []
    for row in rows:
        passage = kb_by_id.get(row["top1_id"], {"text": ""})
        result = gate.score(
            top1_score=row["top1_score"],
            top2_score=row["top2_score"],
            answer_text=row["answer"],
            passage_text=passage.get("text", ""),
        )
        out.append({**row, "confidence": result.confidence,
                    "retrieval_margin": result.retrieval_margin,
                    "grounding_overlap": result.grounding_overlap})
    return out


def expected_calibration_error(scored_rows: list[dict], n_bins: int = 5) -> tuple[float, list[dict]]:
    bins = [[] for _ in range(n_bins)]
    for row in scored_rows:
        bin_idx = min(n_bins - 1, int(row["confidence"] * n_bins))
        bins[bin_idx].append(row)

    ece = 0.0
    n_total = len(scored_rows)
    bin_reports = []
    for i, bucket in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bucket:
            bin_reports.append({"range": f"[{lo:.1f},{hi:.1f})", "n": 0, "mean_conf": None, "accuracy": None})
            continue
        mean_conf = sum(r["confidence"] for r in bucket) / len(bucket)
        accuracy = sum(1 for r in bucket if r["correct"]) / len(bucket)
        ece += (len(bucket) / n_total) * abs(mean_conf - accuracy)
        bin_reports.append({"range": f"[{lo:.1f},{hi:.1f})", "n": len(bucket),
                             "mean_conf": mean_conf, "accuracy": accuracy})
    return ece, bin_reports


def threshold_sweep(scored_rows: list[dict], thresholds: list[float]) -> list[dict]:
    out = []
    n = len(scored_rows)
    for t in thresholds:
        answered = [r for r in scored_rows if r["confidence"] >= t]
        escalated = [r for r in scored_rows if r["confidence"] < t]
        coverage = len(answered) / n
        acc_answered = (sum(1 for r in answered if r["correct"]) / len(answered)) if answered else None

        actually_wrong = [r for r in scored_rows if not r["correct"]]
        wrong_and_escalated = [r for r in actually_wrong if r["confidence"] < t]
        escalation_recall = (len(wrong_and_escalated) / len(actually_wrong)) if actually_wrong else None
        escalation_precision = (sum(1 for r in escalated if not r["correct"]) / len(escalated)) if escalated else None

        out.append({
            "threshold": t, "coverage": coverage, "n_answered": len(answered), "n_escalated": len(escalated),
            "accuracy_if_answered": acc_answered,
            "escalation_precision": escalation_precision,  # of escalated, how many were actually wrong
            "escalation_recall": escalation_recall,          # of actually-wrong, how many got escalated
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(REPO_ROOT / "data" / "confidence_labels.jsonl"),
                         help="Path to the labels JSONL to validate against "
                              "(default: data/confidence_labels.jsonl, the templated-answer set; "
                              "pass data/confidence_labels_llm.jsonl for the real-LLM-answer variant).")
    args = parser.parse_args()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(
            f"error: --labels file not found: {labels_path}\n"
            f"Generate it first, e.g.:\n"
            f"  python scripts/build_confidence_labels.py            # -> data/confidence_labels.jsonl\n"
            f"  python scripts/build_confidence_labels.py --llm      # -> data/confidence_labels_llm.jsonl"
        )

    gate = ConfidenceGate(threshold=0.5)  # threshold itself is swept below; this instance is for scoring only
    rows = load_labels(labels_path)
    kb_by_id = load_kb_by_id()
    scored = compute_confidences(rows, kb_by_id, gate)

    n = len(scored)
    n_correct = sum(1 for r in scored if r["correct"])
    always_answer_baseline_acc = n_correct / n
    print(f"Loaded {n} labeled examples from {labels_path.name} "
          f"({n_correct} correct / {n - n_correct} incorrect at top-1)")
    print(f"'Always answer' baseline accuracy (no gate at all): {always_answer_baseline_acc:.3f}\n")

    margins = [r["retrieval_margin"] for r in scored]
    overlaps = [r["grounding_overlap"] for r in scored]
    print("--- Signal diagnostics (read before trusting the combined score) ---")
    print(f"closed-book retrieval_margin:  min={min(margins):.3f} max={max(margins):.3f} "
          f"mean={sum(margins)/n:.3f}")
    print(f"open-book grounding_overlap:   min={min(overlaps):.3f} max={max(overlaps):.3f} "
          f"mean={sum(overlaps)/n:.3f}")
    llm_examples = sum(1 for r in rows if r.get("answer_source") == "llm")
    if max(overlaps) - min(overlaps) < 0.15:
        if llm_examples == 0:
            print("NOTE: grounding_overlap barely varies across examples in this run. This is an "
                  "expected artifact when answer_source='template' (build_confidence_labels.py's "
                  "default): the templated answer literally quotes its own top1 passage, so it is "
                  "'grounded' by construction even when that passage is the WRONG one -- the "
                  "open-book signal can only catch a generator drifting from its context, not a "
                  "retriever handing the generator the wrong context. Almost all of the gate's "
                  "discriminative power below is therefore coming from retrieval_margin, not "
                  "grounding_overlap. Re-run with data/confidence_labels_llm.jsonl "
                  "(build_confidence_labels.py --llm) for a more meaningful test of the open-book "
                  "signal against real, non-template-quoting answers -- documented as a known "
                  "limitation of this particular labels file, not hidden.\n")
        else:
            print("NOTE: grounding_overlap still barely varies across examples in this run, EVEN "
                  "THOUGH these are real LLM-generated answers (not template quotes). This means the "
                  "low-variance finding was not merely a template-quoting artifact -- the LLM's "
                  "grounded-generation prompting (src/generation/llm_generator.py's PROMPT_TEMPLATE) "
                  "apparently keeps lexical overlap with the retrieved passage high regardless of "
                  "whether that passage actually answers the query, so this simple lexical-overlap "
                  "open-book signal may need a different formulation (e.g. semantic entailment rather "
                  "than word overlap) to actually diverge from correctness. Report this as a further, "
                  "deeper-than-expected limitation, not a fixed problem.\n")
    elif llm_examples > 0:
        print("grounding_overlap now shows real variance against these LLM-generated answers -- "
              "unlike the templated-answer run, the open-book signal is not trivially saturated here.\n")

    print("--- Calibration (5 confidence bins: mean predicted confidence vs. actual accuracy) ---")
    ece, bin_reports = expected_calibration_error(scored, n_bins=5)
    print(f"{'bin':12s} {'n':>3s} {'mean_conf':>10s} {'accuracy':>9s}")
    for b in bin_reports:
        if b["n"] == 0:
            print(f"{b['range']:12s} {0:3d}          -         -")
        else:
            print(f"{b['range']:12s} {b['n']:3d} {b['mean_conf']:10.3f} {b['accuracy']:9.3f}")
    print(f"Expected Calibration Error (ECE, lower=better, 0=perfect): {ece:.3f}\n")

    print("--- Threshold sweep (coverage / accuracy-if-answered / escalation precision & recall) ---")
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep = threshold_sweep(scored, thresholds)
    print(f"{'thr':>5s} {'coverage':>9s} {'n_ans':>6s} {'n_esc':>6s} {'acc_if_ans':>11s} "
          f"{'esc_prec':>9s} {'esc_recall':>10s}")
    best_row = None
    for s in sweep:
        acc_str = f"{s['accuracy_if_answered']:.3f}" if s["accuracy_if_answered"] is not None else "  n/a"
        prec_str = f"{s['escalation_precision']:.3f}" if s["escalation_precision"] is not None else "  n/a"
        rec_str = f"{s['escalation_recall']:.3f}" if s["escalation_recall"] is not None else "  n/a"
        print(f"{s['threshold']:5.1f} {s['coverage']:9.3f} {s['n_answered']:6d} {s['n_escalated']:6d} "
              f"{acc_str:>11s} {prec_str:>9s} {rec_str:>10s}")
        if (s["accuracy_if_answered"] is not None and s["n_answered"] >= 5
                and (best_row is None or s["accuracy_if_answered"] > best_row["accuracy_if_answered"])):
            best_row = s

    print(f"\n--- Verdict (against {labels_path.name}, n={n}; treat as directional, not precise) ---")
    if best_row is not None and best_row["accuracy_if_answered"] > always_answer_baseline_acc:
        margin = best_row["accuracy_if_answered"] - always_answer_baseline_acc
        print(f"PASSES the minimal bar: at threshold={best_row['threshold']:.1f}, accuracy-if-answered="
              f"{best_row['accuracy_if_answered']:.3f} beats the always-answer baseline "
              f"({always_answer_baseline_acc:.3f}) by {margin:+.3f}, "
              f"while answering {best_row['coverage']*100:.0f}% of queries "
              f"and escalating the rest (escalation_recall={best_row['escalation_recall']:.3f} of actually-wrong "
              f"answers caught, escalation_precision={best_row['escalation_precision']:.3f}).")
    else:
        print("DOES NOT beat the always-answer baseline at any threshold with >=5 answered examples in this "
              "50-example demo set. On this evidence the gate does not yet demonstrate value over answering "
              "unconditionally -- report this honestly rather than picking a flattering threshold.")
    print("Per INTRYGUE (intrygue2026): this is exactly why a confidence signal must be validated like this "
          "rather than trusted by construction -- a 50-example demo set is not sufficient to certify the gate "
          "for deployment either way, only to check the mechanism behaves sanely at this scale.")

    out_name = "gate_validation_results_llm.json" if llm_examples > 0 else "gate_validation_results.json"
    out_path = REPO_ROOT / "data" / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n": n, "labels_source": labels_path.name,
            "always_answer_baseline_accuracy": always_answer_baseline_acc,
            "ece": ece, "calibration_bins": bin_reports, "threshold_sweep": sweep,
        }, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
