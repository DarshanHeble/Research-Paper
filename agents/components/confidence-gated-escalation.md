# Component 3 — Confidence-Gated Escalation

Supports contribution 3 (confidence-gated architecture) and underlies the
safety framing used throughout the paper (incorrect agricultural advice can
damage a season's crop).

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
