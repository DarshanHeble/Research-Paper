# Component 1 — Speech-Native Retrieval

Answers **RQ1**: Does speech-native retrieval retain its advantage over
ASR-cascaded retrieval under dialect variation and domain-critical rare
vocabulary?

## What it is

Retrieving relevant documents/passages directly from a speech representation
of the farmer's query, without an intermediate ASR transcription step. This
removes the specific failure mode where ASR mis-transcribes a rare,
domain-critical term (pesticide/pest/disease name), which then propagates
into a bad retrieval and a bad answer.

## Prior art this builds on (all in `reference.bib` / `papers/`)

- **SpeechRAG** (`min2025speechrag`, ICASSP 2025, Amazon) — fine-tunes an
  adapter projecting speech-encoder representations into the embedding space
  of a *frozen* text retriever. Matches/exceeds ASR-then-text-retrieval on
  spoken QA benchmarks. Gap: not evaluated under dialect variation or
  domain-critical rare entities; not agriculture.
- **Google S2R** (`google2025s2r`, production Voice Search) — analogous
  dual-encoder mapping, live in production, accompanied by the Simple Voice
  Questions benchmark (17 languages). Same gap as SpeechRAG.
- **wav2vec 2.0** (`baevski2020wav2vec`) and **HuBERT** (`hsu2021hubert`) —
  self-supervised speech representation backbones underlying both of the
  above.
- **SeamlessM4T** (`barrault2023seamlessm4t`) — multilingual speech
  translation foundation model, a candidate source of pretrained multilingual
  speech encoders for the adapter approach (useful given the dialect target
  is unlikely to have dedicated pretraining data).
- **"Textless NLP" / SpidR** (`spidr2025`) — broader evidence that
  representations optimized for ASR accuracy aren't necessarily optimal for
  downstream spoken-language tasks, which is the theoretical justification
  for why speech-native retrieval can beat the cascade at all.

## The gap this paper targets

Neither SpeechRAG nor S2R has been evaluated:
1. under **dialect variation** (regional/colloquial pronunciation and
   phrasing), or
2. on **domain-critical rare-entity retrieval** (the exact category — pest,
   disease, pesticide names — where ASR error rates are worst, per
   `benchmarkingasr2026`).

Both conditions are exactly the ones smallholder farmers' real queries will
hit, and they compound: a rare entity spoken in a dialect the ASR/encoder
wasn't trained on is the worst case for a cascaded pipeline, and the
untested case for existing speech-native systems.

## Design implications for the architecture

- The adapter/encoder pair should be evaluated specifically on a rare-entity
  subset and a dialect subset of any test set, not just aggregate accuracy —
  aggregate metrics can hide exactly the failure this paper cares about (this
  mirrors the agriculture-weighted WER metric proposed in
  `benchmarkingasr2026` for ASR evaluation; an analogous weighting should
  apply to retrieval evaluation).
- Adapter training is the one step in the whole pipeline allowed to use cloud
  GPU (per contribution 4/scoping) — inference must run offline afterward.
- Because dialect-specific speech corpora barely exist for the target dialect
  (see `dialectmatters2026`, `singh2023respin` — coverage exists for some
  Indian languages/dialects like Chhattisgarhi and Garhwali but not most),
  the adapter likely needs to be trained on a related, better-resourced
  language/dialect and evaluated for transfer, or fine-tuned on a small
  amount of dialect data collected as part of this work.
