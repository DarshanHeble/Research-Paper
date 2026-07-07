# Project Implementation — The System, Not the Paper

This file describes the actual software system the paper proposes, as
something that would need to be built and run to answer RQ1–RQ4. **No code
currently exists in this repository** — it contains only the paper
(`main.tex`) and its bibliography. Treat this file as an implementation plan
to hand to whoever starts building the system, and as the reference an agent
should use to keep the paper's architecture section (`agents/paper.md`,
§III) consistent with what's actually buildable.

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

## Suggested build order (not yet started)

1. Stand up a text-only hybrid (BM25 + dense) retriever over an agricultural
   knowledge base as the baseline — this validates RQ4's premise before any
   speech component exists.
2. Add the ASR-cascaded baseline (Whisper or a comparable model) so
   ASR-cascaded vs. hybrid-text retrieval can be compared (partial RQ1).
3. Fine-tune a speech-to-retrieval adapter (SpeechRAG/S2R-style) on top of a
   pretrained speech encoder, evaluated against the same retrieval baseline
   under dialect variation (completes RQ1).
4. Build the dialect-to-entity mapping layer from public lexicons/KGs (RQ2).
5. Add confidence scoring + escalation gate, validated against labeled
   correct/incorrect examples — not assumed reliable by construction (see
   `intrygue2026` in the lit review on why naive entropy signals misfire).
6. Quantize and package the full pipeline for offline, consumer-hardware
   deployment; measure latency/accuracy tradeoffs (RQ3).

## Constraints carried over from the paper's scoping

- No new dialect speech corpus collection — the dialect-mapping layer must be
  bootstrapped from existing public resources (lexicons, KGs, written corpora).
- Cloud GPU use limited to one-time adapter training; inference must be
  offline-capable.
- Target hardware class: consumer-grade, GPU-constrained (see
  `agents/components/offline-deployment.md` for concrete quantization/RAM
  figures drawn from the literature review).
