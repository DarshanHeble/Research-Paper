# Component 3 — Confidence-Gated Escalation

Supports contribution 3 (confidence-gated architecture) and underlies the
safety framing used throughout the paper (incorrect agricultural advice can
damage a season's crop).

## Implementation status

Implemented at `implementation/src/confidence/{gate,validate_gate}.py`,
combining a closed-book signal (retrieval score margin) and an open-book
signal (lexical grounding overlap between the answer and its source
passage), and validated against 50 labeled examples exactly per the
Section IV-E protocol this doc calls for. First real result: the gate clears
the minimal bar (87.5% accuracy on the ~32% of queries it chooses to answer,
vs. 78.0% unconditional accuracy, catching 81.8% of actually-wrong answers
via escalation) — but validation also **surfaced a real instance of the
INTRYGUE warning below**: the open-book signal barely varies across examples
in this run (because the templated answers quote their own top-1 passage
verbatim, so they're "grounded" even when that passage is wrong), meaning
almost all of the gate's discriminative power currently comes from the
retrieval-margin signal alone, not the intended combination. This is the
component doing exactly what Section IV-E's validation step is for: catching
a signal that looks reasonable in aggregate but is silently carried by only
one input, discoverable only by validating against labels.

**Follow-up re-validation (`build_confidence_labels.py --llm`,
`validate_gate.py --labels data/confidence_labels_llm.jsonl`):** re-labeled
the same 50 queries with real, non-template-quoting Qwen2.5-0.5B-Instruct
answers instead of the template, keeping the correctness rule identical.
Result: the open-book signal is genuinely discriminative once the
template-quoting artifact is removed (grounding_overlap ranges 0.000-1.000
vs. the templated set's 0.868-0.937), and the gate's overall performance
*improves* (94.1% accuracy at 34% coverage, +0.161 over the 0.780 baseline,
vs. the templated set's +0.095; escalation recall 0.909 vs. 0.818). This
traces the item above to an artifact of the first validation set's
construction rather than a fundamental flaw in the signal design — a real,
reproducible (identical across two independent runs, deterministic
generation) positive finding, though still only a 50-example demo-scale
check that should be re-validated again at larger scale before deployment.
See `implementation/README.md` §3 for the full numbers and calibration
table, and §3.5 for the re-validation.

## What it is

A gate that scores the pipeline's confidence in a candidate answer and, below
threshold, escalates the query to a human expert instead of returning a
possibly-wrong answer. Distinct from retrieval quality — this is about
knowing *when the system doesn't know*, not about improving retrieval itself.

## Prior art this builds on

- **Dependable RAG** (`dependablerag2026`) — categorizes hallucination
  detection into *closed-book* methods (internal model signals — token
  probabilities, self-consistency, activation-based signals) and *open-book*
  methods (compare model behavior with vs. without retrieved context — a
  large divergence suggests the model isn't actually grounded in what was
  retrieved).
- **Confidence-based response abstinence** (`abstinence2025`) — argues
  explicitly for abstention in high-stakes RAG (clinical, financial) where a
  confident wrong answer costs more than an abstention. Direct analogy to
  agricultural advice with real crop-damage consequences.
- **INTRYGUE** (`intrygue2026`) — critical caution: naive entropy-based
  confidence signals **misfire in both directions** — flagging correct,
  well-grounded answers as uncertain due to interactions between attention
  mechanisms and entropy computation. This is the single most important
  citation for this component: it means a confidence gate cannot be assumed
  reliable by construction and must be validated empirically.

## The gap / risk this paper must not repeat

Most systems in the literature review (Farmer.Chat, KrishokBondhu,
SukhaRakshak AI, A.A.H.A.R., Krishi Sathi) **answer regardless of
confidence**. None of them gate on confidence at all. This paper's
contribution is adding that gate — but per INTRYGUE, a naive implementation
risks being *worse* than no gate (miscalibrated abstention on correct
answers, or false confidence on wrong ones).

## Design implications for the architecture

- The confidence signal should combine both closed-book and open-book
  approaches per `dependablerag2026`'s taxonomy, rather than relying on a
  single entropy-style score — cross-validated against the INTRYGUE failure
  mode.
- **Must be validated against labeled correct/incorrect examples** before
  being trusted — this validation step belongs explicitly in the
  experimental protocol (§IV), not skipped as an assumed-solved problem.
- Threshold-setting is itself a design decision with a precision/recall
  tradeoff: too permissive → confidently wrong advice reaches farmers; too
  conservative → excessive escalation defeats the point of automation.
  Report this tradeoff curve, don't pick one threshold and hide the
  alternative.
- Escalation target ("a human expert") implies a real operational dependency
  — an extension worker or call center, similar to KrishokBondhu's
  phone-call model — that should be named as a system requirement, not left
  implicit.
