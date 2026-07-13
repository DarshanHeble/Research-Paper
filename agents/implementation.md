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
  the trained speech-native adapter reaches 61.1% top-1 accuracy on its own
  held-out split but ~2% (effectively chance) on the differently-phrased eval
  set (a real generalization gap, not a bug); and the confidence gate's
  open-book signal turned out to carry almost no discriminative power in this
  run because the templated answers quote their own retrieved passage
  verbatim (a concrete, discovered instance of the `intrygue2026` warning
  that confidence signals can look reasonable while being carried by only one
  input).
- Real numbers are not the same as the paper's actual research findings —
  see `implementation/README.md`'s "central honesty caveat" section on why
  every audio-involving number here is measured on `espeak-ng` TTS synthesis,
  not recorded dialect speech, and should be read accordingly.
- These numbers, and the whole prototype, have now been independently
  reproduced end to end on a second machine (Arch Linux, same RTX 3050
  Laptop 6GB GPU class) in a later session, with the same qualitative story
  on every metric — see `implementation/README.md`'s "provenance note" at the
  top of its Results section. `main.tex` Section VI now reports these numbers
  directly in the paper itself, not just in this implementation doc.

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
   real discovered flaw that a follow-up validation then traced to an
   artifact, not a fundamental problem** — `implementation/src/confidence/{gate,
   validate_gate}.py`; clears the minimal "beats always-answer" bar
   (87.5% vs. 78.0% accuracy-if-answered at threshold 0.5) but the first
   validation surfaced that the open-book signal isn't pulling its weight in
   that run — exactly the kind of thing Section IV-E's validation
   requirement is *for* catching. A second validation
   (`build_confidence_labels.py --llm` + `validate_gate.py --labels
   data/confidence_labels_llm.jsonl`) re-built the label set with real,
   non-template-quoting LLM answers and found the signal genuinely
   discriminative (0.000-1.000 range vs. the templated set's 0.868-0.937)
   and the gate's overall performance *improved* (94.1% accuracy at 34%
   coverage, +0.161 over baseline, vs. the templated set's +0.095) — see
   `implementation/README.md` §3.5.
6. ~~Quantize and package the full pipeline~~ **Done for latency; quantization
   now benchmarked too, with a surprising negative result** —
   `implementation/scripts/benchmark_latency.py` measures real
   stage-by-stage latency on an actual RTX 3050 6GB laptop (~1.4-1.5GB peak
   VRAM, all models loaded, reproduced on two separate machines of this class)
   with the LLM (Qwen2.5-0.5B-Instruct) in fp16. `scripts/benchmark_quantization.py`
   (via `LocalLLMGenerator(quantization="int8"|"int4")`, `bitsandbytes`)
   measured INT8/INT4 loading of that same model on the same GPU: both
   *increase* per-query latency substantially (INT8 ~5-6x slower, INT4
   ~40-50% slower) despite reducing peak VRAM as expected — the
   `compactllm2026`-style throughput improvements do not hold for this
   small model on this consumer GPU class, contrary to the naive
   expectation. See `implementation/README.md` §4.5.

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
- **Quantization latency regression, unexplained beyond a plausible
  mechanism.** Whether the INT8/INT4 slowdown found here reverses at larger
  model scale or on datacenter-class GPUs with dedicated low-precision
  tensor cores is untested — a narrower, more concrete question than the
  original "quantization not yet implemented" gap.
- **The human-expert escalation target** is a stubbed string, not a real
  operational integration (phone queue, ticketing system, etc.).
- **Confidence-gate re-validation at larger scale.** The gate's open-book
  signal was re-validated against real LLM answers and found genuinely
  discriminative (see item 5 above), but only on the same 50-example
  demo-scale, synthetic-TTS-derived set — this should be re-checked again at
  larger scale and on real queries before being trusted operationally.

## Constraints carried over from the paper's scoping

- No new dialect speech corpus collection — the dialect-mapping layer must be
  bootstrapped from existing public resources (lexicons, KGs, written corpora).
- Cloud GPU use limited to one-time adapter training; inference must be
  offline-capable.
- Target hardware class: consumer-grade, GPU-constrained (see
  `agents/components/offline-deployment.md` for concrete quantization/RAM
  figures drawn from the literature review).
