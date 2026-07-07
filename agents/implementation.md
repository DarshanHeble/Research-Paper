# Project Implementation — The System, Not the Paper

This file describes the actual software system the paper proposes. Treat it
as the reference an agent should use to keep the paper's architecture section
(`agents/paper.md`, §III) consistent with what's actually buildable.

## Status: a runnable prototype exists — read `implementation/README.md` first

A best-effort, genuinely runnable implementation of every component below
lives in `implementation/` at the repo root. **`implementation/README.md` is
the authoritative status report** — it documents, with real pasted output,
what's real vs. demo-scale vs. stubbed, and should be read before citing any
number from it anywhere else (including in the paper). The summary:

- All four pipeline stages are implemented and wired into one pipeline
  (`implementation/src/pipeline.py`), with 24 passing tests
  (`implementation/tests/`, independently reverified).
- Everything runs on real (if small, hand-curated, clearly-labeled) demo
  data — a 42-passage KB, a 29-entry dialect lexicon, 50 stratified eval
  queries — **not** the real target-dialect speech corpus the paper's own
  evaluation protocol (Section IV) calls for, because that corpus does not
  exist yet (this was true when the paper was written and remains true).
- Headline real result: dialect-to-entity mapping is what makes dialectal
  queries retrievable at all on this demo set (recall@1 on dialectal strata
  goes from ~0.0 to 0.615 once the mapping layer is applied) — a real,
  measured, clean confirmation of the RQ2/RQ4 premise on this demo scale,
  with the caveat that it needs re-testing at a larger, messier KB scale.
- Two honest negative/partial results worth knowing before reading further:
  the trained speech-native adapter reaches 55.6% top-1 accuracy on its own
  held-out split but ~0% on the differently-phrased eval set (a real
  generalization gap, not a bug); and the confidence gate's open-book signal
  turned out to carry almost no discriminative power in this run because the
  templated answers quote their own retrieved passage verbatim (a concrete,
  discovered instance of the `intrygue2026` warning that confidence signals
  can look reasonable while being carried by only one input).
- Real numbers are not the same as the paper's actual research findings —
  see `implementation/README.md`'s "central honesty caveat" section on why
  every audio-involving number here is measured on `espeak-ng` TTS synthesis,
  not recorded dialect speech, and should be read accordingly.

The rest of this file is the original architecture/build-order reference —
still accurate as a description of intent, now cross-linked to what's
actually been built against each point.

## System overview

A pipeline that takes a spoken farmer query in a low-resource dialect and
returns an agricultural advisory answer, with four pipeline stages
corresponding to the paper's four contributions:

```
[Farmer speech query]
        │
        ▼
┌───────────────────────┐
│ 1. Speech Encoder      │  wav2vec 2.0 / HuBERT / SeamlessM4T encoder,
│    + Retrieval Adapter │  fine-tuned to project speech into a frozen
│    (speech-native      │  text-retriever embedding space (SpeechRAG-/
│    retrieval)          │  S2R-style dual-encoder alignment)
└──────────┬────────────┘
           │  speech embedding
           ▼
┌───────────────────────┐
│ 2. Dialect-to-Entity   │  Maps colloquial/regional terms (or their
│    Mapping Layer       │  embeddings) onto standard scientific entities
│                        │  (crop/pest/disease/pesticide names) using a
│                        │  bootstrapped lexicon + KG lookup, not new
│                        │  dialect speech collection
└──────────┬────────────┘
           │  normalized query representation
           ▼
┌───────────────────────┐
│ 3. Hybrid Retriever    │  BM25 (lexical) + dense retriever, fused —
│    (lexical + dense)   │  lexical wins on rare exact entities, dense
│                        │  wins on paraphrase; hybrid combines both
└──────────┬────────────┘
           │  retrieved passages + retrieval confidence
           ▼
┌───────────────────────┐
│ 4a. Confidence Gate    │──low confidence──▶  Escalate to human expert
└──────────┬────────────┘
           │ high confidence
           ▼
┌───────────────────────┐
│ 4b. Local LLM          │  Quantized, offline-capable generator producing
│     Generation + TTS   │  the grounded answer, delivered as speech
└───────────────────────┘
```

Everything below the speech encoder must run **fully offline** on
consumer-grade hardware (RQ3) — cloud GPU is reserved only for the one-time
adapter fine-tuning step (contribution 4).

## Component breakdown (see `agents/components/*.md` for depth on each)

1. **Speech-native retrieval** — `agents/components/speech-native-retrieval.md`
2. **Dialect-to-entity mapping** — `agents/components/dialect-entity-mapping.md`
3. **Confidence-gated escalation** — `agents/components/confidence-gated-escalation.md`
4. **Offline/edge deployment** — `agents/components/offline-deployment.md`

Retrieval strategy (hybrid lexical+dense, RQ4) cuts across components 1 and
2 rather than being its own pillar — it's the retrieval backbone that both
the speech embeddings and the dialect-mapped entities feed into.

## Build order — status against each step

1. ~~Stand up a text-only hybrid (BM25 + dense) retriever~~ **Done** —
   `implementation/src/retrieval/{bm25,dense,hybrid}_retriever.py`, real
   Reciprocal Rank Fusion over the demo KB.
2. ~~Add the ASR-cascaded baseline~~ **Done** —
   `implementation/src/asr/whisper_cascade.py` (`faster-whisper` tiny), real
   transcripts in `implementation/data/asr_cascade_transcripts.json`.
3. ~~Fine-tune a speech-to-retrieval adapter~~ **Done, as a proof of
   mechanism** — `implementation/src/speech_retrieval/{adapter,train_adapter,
   speech_retriever}.py`; trains for real in ~20s on synthetic TTS audio, but
   does not generalize past its own training distribution (see status
   section above) — the mechanism works, dialect generalization remains the
   paper's open question, unresolved by this prototype as expected.
4. ~~Build the dialect-to-entity mapping layer~~ **Done, mechanism real,
   lexicon demo-scale** — `implementation/src/dialect_mapping/mapper.py`,
   29 self-curated entries (see `implementation/data/dialect_lexicon.json`'s
   `_readme` field) — not the domain-expert/public-KG-sourced resource a real
   deployment needs per contribution 2, but a working lookup-and-normalize
   mechanism.
5. ~~Add confidence scoring + escalation gate~~ **Done, validated, with a
   real discovered flaw** — `implementation/src/confidence/{gate,
   validate_gate}.py`; clears the minimal "beats always-answer" bar
   (87.5% vs. 78.0% accuracy-if-answered at threshold 0.5) but validation
   itself surfaced that the open-book signal isn't pulling its weight in
   this run — exactly the kind of thing Section IV-E's validation
   requirement is *for* catching.
6. ~~Quantize and package the full pipeline~~ **Done for latency; not
   quantized** — `implementation/scripts/benchmark_latency.py` measures real
   stage-by-stage latency on the actual RTX 3050 6GB laptop this was built
   on (~1.48GB peak VRAM, all models loaded); the LLM (Qwen2.5-0.5B-Instruct)
   runs in fp16, not INT4/INT8-quantized — quantization itself is not yet
   implemented, so the `compactllm2026`-style throughput figures in
   `agents/components/offline-deployment.md` remain uncompared against a
   quantized version of this exact pipeline.

## What's still genuinely open after this prototype

- **Real dialect speech data.** Nothing here changes the fact that no
  target-dialect speech corpus exists — this remains the central blocker for
  actually answering RQ1 and RQ2 as the paper poses them, not something a
  software prototype can solve on its own.
- **Embedding-space dialect mapping** (architecture #2 in
  `agents/components/dialect-entity-mapping.md`) — the prototype only
  implements the simpler post-transcription lexicon-lookup version, so
  dialect mapping and speech-native retrieval don't currently compose (see
  `implementation/src/pipeline.py`'s module docstring).
- **Quantization** of the local LLM for the offline deployment story (RQ3).
- **The human-expert escalation target** is a stubbed string, not a real
  operational integration (phone queue, ticketing system, etc.).
- **Confidence-gate signal quality** — the open-book grounding signal needs
  re-validating against non-template-quoting LLM answers before being
  trusted; see `implementation/README.md` §3 for the full finding.

## Constraints carried over from the paper's scoping

- No new dialect speech corpus collection — the dialect-mapping layer must be
  bootstrapped from existing public resources (lexicons, KGs, written corpora).
- Cloud GPU use limited to one-time adapter training; inference must be
  offline-capable.
- Target hardware class: consumer-grade, GPU-constrained (see
  `agents/components/offline-deployment.md` for concrete quantization/RAM
  figures drawn from the literature review).
