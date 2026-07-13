# `main.tex` — Paper Structure and Writing Status

Owner file for anything touching the paper's prose, structure, or LaTeX.

## Document class and packages

`\documentclass[conference]{IEEEtran}` — standard IEEE conference two-column
format. Packages in use: `cite`, `amsmath,amssymb,amsfonts`, `algorithmic`,
`graphicx`, `textcomp`, `xcolor`, `booktabs`, `tabularx`, `url`. Bibliography
style is `IEEEtran` via `\bibliographystyle{IEEEtran}` / `\bibliography{reference}`.

Do not remove `\raggedbottom` or change the class options without a reason —
they're the IEEE-conference defaults this template ships with.

## Title, author, venue framing

- Title: *Speech-Native Retrieval-Augmented Generation for Agricultural
  Advisory in Low-Resource Dialects*
- Author: Darshan Pundlik Heble, PG Scholar, Dept. of Computer Science,
  Christ (Deemed to be University), Bangalore, India.
- Single-author, single-affiliation. The commented-out `\and` block is a
  template leftover for a second author — leave commented unless a co-author
  is actually added.

## Research questions (RQ1–RQ4) — the spine of the paper

1. **RQ1**: Does speech-native retrieval retain its advantage over
   ASR-cascaded retrieval under dialect variation and domain-critical rare
   vocabulary?
2. **RQ2**: Does a dialect-to-scientific-entity mapping layer improve
   retrieval accuracy for queries phrased in regional/colloquial terms?
3. **RQ3**: Can the pipeline operate fully offline, on resource-constrained
   consumer hardware, at acceptable latency and accuracy?
4. **RQ4**: Does hybrid (lexical + dense) retrieval combined with dialect
   mapping outperform either method alone on queries containing
   domain-critical rare entities?

Every architecture decision, experiment, and evaluation metric introduced
later in the paper should be traceable to one of these four.

## Four claimed contributions

1. Identifying the specific gap: dialect variation + domain-critical rare
   vocabulary, unaddressed by existing speech-native retrieval systems
   (SpeechRAG, S2R).
2. A dialect-to-scientific-entity mapping layer bootstrapped from low-cost
   public resources (not new dialect speech collection).
3. A confidence-gated RAG architecture that escalates low-confidence queries
   to a human expert.
4. A system design + experimental protocol scoped to consumer-grade,
   GPU-constrained hardware (cloud GPU reserved only for one-time adapter
   training).

## Section-by-section status

| § | Title | Status |
|---|---|---|
| I | Introduction | **Done.** Motivation, gap analysis, RQ1–4, 4 contributions, roadmap paragraph. |
| II | Literature Review | **Done.** 8 subsections: AI-driven agri advisory, speech processing for low-resource Indic dialects, speech-native retrieval, retrieval strategy (keyword/dense/hybrid), dialect-aware KR, confidence estimation, offline/edge deployment, synthesis. Well-cited — treat as the citation-density bar for new sections. |
| III | Proposed Architecture | Drafted — see below. |
| IV | Experimental Protocol / Methodology | Drafted — see below. |
| V | Evaluation Plan / Expected Outcomes | Drafted — deliberately **not** a "Results" section with real numbers on the target dialect (that data doesn't exist). Framed as hypotheses tied to RQ1–4 plus the metrics/protocol that would test them. Still open — Section VI does not resolve these. |
| VI | Prototype Implementation and Preliminary Measurements | **Done, with real numbers.** Reports actual measured results from `implementation/` (retrieval recall@1 table, adapter training curve, confidence-gate calibration/threshold table, latency benchmark), explicitly scoped as a demo-scale mechanism check on synthetic-TTS audio, distinct from Section V's still-open hypotheses on real dialect data. See `agents/implementation.md` and `implementation/README.md` for the source of every number here — never edit a number in this section without re-deriving it from an actual run. |
| VII | Conclusion | Drafted — summary, honest limitations (RQ1–4 still open on target dialect despite the Section VI prototype), future work updated to reference concrete prototype findings (adapter generalization gap, gate signal weakness, embedding-space dialect mapping, quantization). |

### Important structural correction made

The original file had a roadmap paragraph in §I promising "Section III
presents the proposed architecture; Section IV describes the experimental
setup; Section V presents results; Section VI concludes" — but the actual
section headers were leftover template/mismatched content (`COMPARATIVE
ANALYSIS OF PREVIOUS WORKS`, `Performance Analysis` with subsections on
knowledge graphs / multi-agent filtering / domain-specific accuracy — none of
which connect to this paper's own architecture). These were replaced with
sections that actually match the roadmap and the four contributions. If you
find the roadmap paragraph and the section headers disagree again after
future edits, fix the mismatch — don't leave both inconsistent.

## IEEE formatting conventions to preserve

- Numbered citations via `\cite{key}`, numeric/bracketed in output — never
  switch to author-year in-text.
- Section headings: Roman numeral + Title Case, matching the existing
  pattern (`\section{Introduction}`, `\section{Literature Review}`, etc.).
  This was inconsistent (`\section{LITERATURE REVIEW}` was ALL CAPS while
  every other section was Title Case) until a full-document pass fixed it
  on 13 July 2026 — all seven sections are now Title Case. Keep it that way
  for any new section.
- Abstract: single paragraph, no citations inside it (IEEE convention —
  currently followed correctly).
- `\begin{IEEEkeywords}...\end{IEEEkeywords}` block right after the abstract.
- Tables/figures: use the commented-out templates near the bottom of the file
  as the boilerplate for `\caption`, `\label`, `booktabs`-style rules — don't
  introduce a different table style.

## Compiling

Run from inside `paper/` (where `main.tex` and `reference.bib` now live):

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires a working `texlive` install with IEEEtran class available
(`texlive-publishers` or equivalent on most distros). Check for undefined
references/citations in the log after the run — `bibtex main` output listing
"undefined" keys usually means a `reference.bib` key was mistyped or removed.
