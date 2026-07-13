# AGENTS.md — Speech-Native RAG for Agricultural Advisory

This file orients any agent (human or AI) working in this repository. Read this
first, then follow the pointers below to the file relevant to the task at hand.

## What this project is

An IEEE conference paper: **"Speech-Native Retrieval-Augmented Generation for
Agricultural Advisory in Low-Resource Dialects"**, single-authored by Darshan
Pundlik Heble (PG Scholar, Dept. of Computer Science, Christ (Deemed to be
University), Bangalore).

The paper is a **framework / protocol proposal**, not a paper reporting
completed experiments. It argues that no existing system combines four things
for agricultural voice advisory in dialect-rich, low-resource settings:

1. **Speech-native retrieval** — retrieving directly from a speech
   representation, skipping the ASR transcription step that is the dominant
   source of error on domain-critical vocabulary (pesticide/pest/disease
   names).
2. **Dialect-to-scientific-entity mapping** — normalizing colloquial/regional
   farmer terms to standard agricultural nomenclature, bootstrapped from
   low-cost public resources rather than new dialect speech collection.
3. **Confidence-gated escalation** — abstaining and routing to a human expert
   when the system is not confident, rather than always answering.
4. **Fully offline operation** on consumer-grade, GPU-constrained hardware.

It poses four research questions (RQ1–RQ4, see `agents/paper.md`) and proposes
an experimental protocol to answer them. **Section V's hypotheses remain
unanswered on the target dialect** — no real target-dialect speech corpus
exists, and that remains true and deliberate; do not fabricate numbers to
fill that gap. What *does* now exist is a runnable software prototype of the
architecture (`implementation/`, see below) that produces real measured
numbers on small, honestly-labeled demo data, and **Section VI of `main.tex`
itself now reports those numbers** — tables, a training curve, a gate
calibration sweep, a latency benchmark, a confidence-gate re-validation
against real (non-template-quoting) LLM answers, and an LLM quantization
benchmark, explicitly framed as a demo-scale mechanism check distinct from
Section V's still-open hypotheses, not as an execution of the paper's own
protocol. See `agents/paper.md` and `implementation/README.md` for how each
is meant to be read without conflating the two.

## Repository layout

```
paper/             Everything the paper itself needs to compile
  main.tex             IEEE conference paper source (compiles with IEEEtran)
  reference.bib        Bibliography (33 entries), with verification-status tags
  pdfs/                Local copies of every cited reference (PDF or .txt citation)
implementation/    Runnable prototype of the proposed system (see its README.md)
agents/            This documentation set
  paper.md              paper/main.tex: structure, IEEE formatting rules, writing status
  bibliography.md       paper/reference.bib + paper/pdfs/: verification keys, how to add refs
  implementation.md     The proposed system as software; status of the real prototype
  components/           One file per architecture component (deep dives + implementation status)
    speech-native-retrieval.md
    dialect-entity-mapping.md
    confidence-gated-escalation.md
    offline-deployment.md
```

## Which file to read for which task

| Task | Read |
|---|---|
| Editing paper prose, LaTeX, citations, section structure | `agents/paper.md` |
| Adding/verifying a reference, understanding `paper/pdfs/` folder | `agents/bibliography.md` |
| Running or extending the actual prototype | `implementation/README.md` first, then `agents/implementation.md` |
| Deep-diving one of the four architecture pillars (design + implementation status) | `agents/components/*.md` |

## Ground rules for anyone (agent or human) working on this paper

- **Never invent experimental results.** No experiments have been run yet. If
  a section needs numbers that don't exist, either write it as an evaluation
  *plan* (metrics, protocol, hypotheses tied to RQs) or leave an explicit
  `%TODO` placeholder — never silently insert plausible-looking fake data.
- **IEEE format throughout.** This uses the `IEEEtran` conference class
  (`\documentclass[conference]{IEEEtran}`), `\cite{}` keys from `reference.bib`,
  and `IEEEtran` bibliography style. Follow IEEE conference conventions for
  section numbering (Roman numerals, ALL CAPS or Title Case per existing
  style), figure/table placement, and citation style (numeric, bracketed).
- **Every factual/technical claim needs a citation** or must be flagged as the
  paper's own proposed contribution. The literature review is unusually
  well-sourced — match that standard in new sections.
- **Keep the four contributions and RQ1–RQ4 as the spine.** Every new section
  should visibly trace back to one or more of them.
- Compile-check after edits, from inside `paper/`: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` (see `agents/paper.md` for details).
