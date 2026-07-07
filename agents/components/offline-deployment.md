# Component 4 — Offline and Edge Deployment

Answers **RQ3**: Can the pipeline operate fully offline, on
resource-constrained consumer hardware, at acceptable latency and accuracy?
Also underlies contribution 4 (scoping cloud GPU to one-time adapter
training only).

## What it is

Running the entire inference-time pipeline — speech encoder, dialect
mapping, hybrid retrieval, confidence gate, generation, TTS — on
consumer-grade hardware with no live internet dependency, because the rural
settings this system targets frequently lack reliable connectivity.

## Prior art this builds on

- **Compact LLM deployment** (`compactllm2026`) — concrete figures: 4-bit
  quantized Llama-3 8B gives usable but slow inference (2–3 tokens/second) on
  smartphones with ≥6 GB RAM; 8-bit variants need a discrete GPU with ≥8 GB
  VRAM for comparable throughput. These are the numbers to benchmark the
  proposed pipeline against.
- **A.A.H.A.R.** (`dubey2025aahar`) — offline, quantized LLM + ASR fine-tuned
  for agricultural vocabulary/regional accents, sub-2-second response times.
  Proves offline agricultural RAG is achievable for **text-based** systems
  already.
- **Bengali cross-lingual RAG case study** (`hossain2026bengalirag`) — fully
  local, consumer-hardware deployment, sub-20-second end-to-end latency.
  Same caveat: text-based.
- **Tigray offline plant-disease detection** (`tigray2025plantdisease`) —
  demonstrates the general deployment pattern (quantization, on-device
  runtime conversion, dialect-localized interfaces) for a **vision** modality
  — evidence the pattern generalizes across modalities, not evidence it
  already works for speech-native retrieval specifically.

## The open question this paper poses (not yet answered)

Every offline precedent above is either text-only or a different modality.
**Whether offline operation still holds once a speech-native retrieval layer
and a dialect-mapping layer are added on top is explicitly unresolved** — this
is RQ3, not a solved problem being cited for completeness.

## Design implications for the architecture / experimental protocol

- Benchmark end-to-end latency (speech in → answer out) on a defined
  consumer hardware target (e.g., ≥6 GB RAM smartphone class, or a
  GPU-constrained laptop/mini-PC — pick one and state it explicitly, don't
  leave "consumer-grade" vague in §III/§IV).
- Report latency **and** accuracy together, not latency alone — a fast
  offline pipeline that regresses accuracy relative to a cloud-hosted
  cascade doesn't answer RQ3 in the affirmative.
- The added components (speech encoder + adapter, dialect mapping lookup,
  confidence scoring) each add inference-time cost on top of the
  already-benchmarked LLM generation step — budget for all of them in the
  latency measurement, not just the generation step that prior work already
  measured.
- Cloud GPU use should be auditable as a one-time cost (adapter training) vs.
  recurring cost (would violate the offline/consumer-hardware framing) — make
  this distinction explicit in the experimental protocol.
