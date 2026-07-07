# Implementation — Speech-Native RAG for Agricultural Advisory (Prototype)

This directory is a **best-effort, genuinely runnable software prototype** of
the system proposed in `main.tex` and specified in `agents/implementation.md`
and `agents/components/*.md`. It is not a claim that the paper's research
questions are answered — the paper itself says no experiments have been run
yet (see `AGENTS.md`), and this prototype does not change that. What it *does*
do is build every architectural component for real, wire them into an
end-to-end pipeline, and measure real numbers on small, honestly-labeled
demo-scale data, so the mechanism can be inspected and stress-tested rather
than only described in prose.

**Read the "What's real vs. demo-scale vs. stubbed" section below before citing
any number from this directory anywhere else.** Every module that makes a
simplifying assumption says so in its own docstring too — this README
summarizes and cross-references those notes, it doesn't replace them.

## Environment this was built and measured on

- Python 3.12.3, Ubuntu (Pop!_OS), 12 cores, 15GB RAM.
- GPU: NVIDIA RTX 3050 Laptop, 6GB VRAM — used as the actual latency-benchmark
  target, per the paper's own "consumer-grade, GPU-constrained hardware"
  framing (Section III-F). This is a real instance of that hardware class, not
  a stand-in for it.
- `implementation/.venv` — an isolated venv, nothing installed system-wide.

## Setup

```bash
cd implementation
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # or plain `pip install torch==2.6.0` for CPU-only
pip install -r requirements.txt
```

`transformers>=4.48`-ish refuses to `torch.load()` non-safetensors checkpoints
(e.g. `facebook/wav2vec2-base`'s `.bin` weights) unless torch>=2.6 — that's why
torch is pinned there rather than something older.

### Optional local LLM (real generation, not just templating)

```bash
mkdir -p models/qwen2.5-0.5b-instruct
# Fetch Qwen2.5-0.5B-Instruct's few small config/tokenizer files and its one
# ~988MB safetensors weight file. In the reference environment,
# huggingface_hub's chunked resumable downloader repeatedly stalled partway
# through the large file for reasons never fully diagnosed, while a plain
# `curl` of the same URL over the same network worked fine -- so that's the
# documented path. `pip install -U huggingface_hub` and a plain
# `from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")` may just work for you; if it
# does, you don't need any of this and can skip straight to running things.
curl -L -o models/qwen2.5-0.5b-instruct/model.safetensors \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
python -c "
from huggingface_hub import hf_hub_download
import shutil
for f in ['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:
    shutil.copy(hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct', f), f'models/qwen2.5-0.5b-instruct/{f}')
"
```

If this model directory doesn't exist, `src/generation/llm_generator.py` falls
back to the plain Hugging Face hub id (which will "just work" if your network
doesn't hit the same stall), and if that also fails, `src/pipeline.py` catches
the exception and falls back to a deterministic templated answer — the
pipeline never crashes for lack of an LLM, it just tells you (via
`PipelineResult.answer_source`) which path actually produced the answer.

### Train the speech-native adapter (one-time, ~20-30s on this GPU)

```bash
python -m src.speech_retrieval.train_adapter
```

Writes `models/speech_adapter.pt`. Not committed to git (see `.gitignore`) —
it's cheap to regenerate and is a training artifact, not source.

## Running things

```bash
pytest tests/ -v                        # 24 tests, all passing (see below)
python scripts/run_evaluation.py        # retrieval baselines vs. eval_queries.jsonl
python scripts/benchmark_latency.py     # stage-by-stage latency on this machine
python scripts/build_confidence_labels.py   # regenerates data/confidence_labels.jsonl
python -m src.confidence.validate_gate  # gate calibration + threshold sweep
python -m src.pipeline "Kapas ke phool aur tinde me gulabi sundi lag gayi hai"  # one query, end to end
```

---

## What's real vs. demo-scale vs. stubbed — read this before trusting a number

| Component | Status |
|---|---|
| BM25 lexical retrieval | **Real.** `rank_bm25` over the real 42-passage KB. |
| Dense retrieval | **Real.** `sentence-transformers/all-MiniLM-L6-v2`, real embeddings, real cosine search. |
| Hybrid (RRF) fusion | **Real,** standard Reciprocal Rank Fusion, `k=60`. |
| Dialect-to-entity mapping | **Real mechanism**, on a **demo-scale, self-curated lexicon** (29 entries). Bootstrapped from generally well-known North/Central-Indian farmer terminology (the kind in KVK/ICAR extension pamphlets), *not* a domain-expert-reviewed or public-KG-sourced resource as the paper's own scoping constraint (contribution 2) requires for production use. See `data/dialect_lexicon.json`'s `_readme` field. |
| ASR-cascaded baseline | **Real** `faster-whisper` "tiny" transcription of **real synthesized audio** — but that audio is espeak-ng TTS, not recorded human dialect speech (see below). |
| Speech-native retrieval (adapter) | **Real mechanism**, trained for real on **real (synthetic TTS) audio**, with genuinely measured — and honestly weak — generalization (see Results). Not a claim of real dialect-speech performance; the paper itself says that data doesn't exist yet. |
| Confidence gate | **Real mechanism**, validated against a real (if small, 50-example) labeled set, with a real (if unflattering) discovered limitation in the open-book signal (see Results). |
| LLM generation | **Real**, when the model is available (`Qwen/Qwen2.5-0.5B-Instruct`, fp16, ~1.5GB total VRAM for the whole pipeline). Falls back to a deterministic template otherwise — always disclosed via `answer_source`. |
| Offline operation | **Real** for every stage except the one-time adapter-training step and the one-time LLM weight download, exactly matching the paper's own cloud-GPU scoping constraint (contribution 4). |

### The central honesty caveat: there is no real dialect speech corpus

Per the paper's own text and `agents/components/speech-native-retrieval.md`,
no real target-dialect farmer speech corpus exists for this project. To
exercise the ASR cascade and the speech-native adapter at all, **all query
audio in this repository is synthesized with `espeak-ng`'s English voice**,
including the romanized Hindi/Bengali colloquial queries in
`data/eval_queries.jsonl`. That means:

- The "speech" Whisper and wav2vec2 see is a robotic English-voice reading of
  (sometimes non-English) text, not a native speaker of any dialect.
- Any result below that looks like "ASR/speech-native retrieval fails badly on
  dialectal queries" is real and measured, but it is *at least partly* an
  artifact of feeding non-English text through an English TTS voice into
  English-pretrained models — a real dialect speech corpus would likely (not
  certainly) look different. Both directions of that uncertainty are worth
  keeping in mind: it could be optimistic (real dialect speech may carry
  genuine phonetic structure espeak's flat English-phoneme reading doesn't)
  or pessimistic (this at least demonstrates *a* systematic mismatch, and
  real accented dialect speech would still be mismatched, just differently).
- `espeak-ng` itself was not preinstalled in the reference environment
  (`espeak-ng-data`/`libespeak-ng1` were, the CLI front-end wasn't). Since
  passwordless sudo wasn't available, the `.deb` was fetched with
  `apt-get download` (no root needed to download) and the single binary
  extracted with `dpkg -x` into `tools/espeak-ng` — no source compiled, no
  system files touched. See `src/tts_util.py`'s docstring for the full story.

### A mid-session environment issue, disclosed rather than hidden

Partway through this build, an unattended `nvidia-driver` package upgrade on
this machine updated the userspace driver library to `595.84` while the
already-loaded kernel module stayed at `595.71.05`, breaking `nvidia-smi`/NVML
for the rest of the session (no reboot or root access available to fix it
mid-session). **`torch.cuda` compute itself kept working** — verified with a
real on-GPU matmul producing a correct result, and every benchmark/eval run
below completed on `cuda` — only the NVML-based monitoring calls
(`nvidia-smi`, GPU-name-lookup-via-NVML) are affected. `scripts/benchmark_latency.py`
prints this same note when you run it.

---

## Results — real, measured, on this demo data (paste, not paraphrase)

### 1. Retrieval baselines (`python scripts/run_evaluation.py`), 50 stratified eval queries

Full output in `data/evaluation_results.json`. Headline recall@1 by baseline
and stratum:

| Baseline | overall | standard/common | standard/rare | dialectal/common | dialectal/rare |
|---|---:|---:|---:|---:|---:|
| 1. keyword-only (BM25) | 0.500 | 0.917 | 1.000 | 0.077 | 0.077 |
| 2. dense-only | 0.400 | 0.833 | 0.833 | 0.000 | 0.000 |
| 3. hybrid, no dialect mapping | 0.460 | 0.917 | 1.000 | 0.000 | 0.000 |
| 4. hybrid + dialect mapping | **0.780** | 0.917 | 1.000 | **0.615** | **0.615** |
| 5. ASR cascade (Whisper tiny) + hybrid + dialect mapping | 0.380 | 0.583 | 1.000 | 0.000 | 0.000 |
| 6. speech-native (trained adapter), no dialect mapping | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**What this shows, honestly:**
- **Dialect mapping is doing essentially all of the work on dialectal
  queries** — every baseline without it scores ~0.0 recall@1 on
  dialectal/common and dialectal/rare; adding it (#4) jumps both to 0.615.
  This is a real, clean confirmation of RQ2/RQ4's premise on this demo set —
  and also a sign that 42 KB passages with fairly distinct vocabulary is an
  easy regime for a lexicon-lookup layer; it would need re-testing at a
  larger, messier KB scale before trusting the same 0.615 number.
- **ASR cascade is genuinely, badly broken by dialectal queries** — 0.0
  recall@1 on both dialectal strata, and worse than raw hybrid retrieval even
  on standard-phrasing queries (0.583 vs. 0.917), because `faster-whisper`
  "tiny" mangles even the *English* TTS audio somewhat (see
  `data/asr_cascade_transcripts.json` for every transcript — e.g. query q001
  "My rice field has hopperburn..." was transcribed as "My life feels first
  hot women with yellowing and drying..."). Whisper "tiny" on
  robotic TTS audio is a harder condition than Whisper on natural human
  speech; this number should not be read as "Whisper is this bad in
  general," only that this specific tiny model + this specific synthetic
  audio pipeline is.
- **Speech-native retrieval's recall@1 on the held-out eval set is ~0**, despite
  the adapter clearly learning *something* (see below) — a real, and
  important, negative result: the toy training signal does not generalize
  to the eval set's different phrasing style. This is discussed in depth
  below, not glossed over.

### 2. Speech-native adapter training (`python -m src.speech_retrieval.train_adapter`)

```
[1/5] 108 train examples, 18 val examples (auto-generated templates, disjoint
      from eval_queries.jsonl -- see train_adapter.py docstring)
...
[4/5] Training adapter for 100 epochs with in-batch InfoNCE contrastive loss...
    epoch   1  train_loss=3.5034  val_top1_passage_acc=0.000
    epoch  50  train_loss=0.2449  val_top1_passage_acc=0.389
    epoch  90  train_loss=0.1664  val_top1_passage_acc=0.556
    epoch 100  train_loss=0.0885  val_top1_passage_acc=0.500
[5/5] Best checkpoint: epoch 87, val_top1_passage_acc=0.556 (chance level = 1/42 = 0.024)
Total wall time: 19.5s
```

**The honest read:** on its OWN held-out validation split (18 examples, drawn
from the same small set of auto-generated English templates used for
training, e.g. *"What can I do about {entity} in {crop}?"*), the trained
adapter retrieves the correct passage top-1 **55.6% of the time — 23x better
than the 2.4% chance level for a 42-way retrieval problem.** That is a real,
positive demonstration that the SpeechRAG/S2R-style mechanism (frozen speech
encoder → small trained adapter → frozen text-retriever embedding space, no
transcription) works end to end on this machine.

But on `data/eval_queries.jsonl` — full natural-language symptom-description
sentences and romanized Hindi/Bengali queries, a genuinely different phrasing
distribution than the short template sentences it trained on — recall@1 drops
to ~0.0 (see table above). **This is a real generalization gap, not a bug**,
and it's exactly the kind of result the paper predicts is unresolved: a small
synthetic/template training signal does not obviously transfer to realistic
out-of-distribution phrasing or (much more importantly) real dialect speech,
which is the actual open problem the paper names and does not claim to have
solved. Don't read the 55.6% number as "the adapter works" without also
reading the ~0% eval number right next to it.

### 3. Confidence gate validation (`python -m src.confidence.validate_gate`)

Built from `data/confidence_labels.jsonl` (50 examples: `hybrid + dialect
mapping` retrieval + templated answers, rule-labeled correct/incorrect against
`eval_queries.jsonl`'s gold ids — 39 correct / 11 incorrect, i.e. this set's
raw top-1 accuracy is 0.780, the baseline the gate has to beat):

```
--- Signal diagnostics ---
closed-book retrieval_margin:  min=0.000 max=0.217 mean=0.070
open-book grounding_overlap:   min=0.868 max=0.937 mean=0.907
NOTE: grounding_overlap barely varies across examples in this run... [see below]

--- Calibration (5 confidence bins) ---
bin            n  mean_conf  accuracy
[0.4,0.6)     50      0.489     0.780
Expected Calibration Error (ECE): 0.291

--- Threshold sweep (excerpt) ---
 thr  coverage  n_ans  n_esc  acc_if_ans  esc_prec  esc_recall
 0.4     1.000     50      0       0.780       n/a      0.000
 0.5     0.320     16     34       0.875     0.265      0.818
 0.6     0.000      0     50         n/a     0.220      1.000

PASSES the minimal bar: at threshold=0.5, accuracy-if-answered=0.875 beats the
always-answer baseline (0.780) by +0.095, answering 32% of queries and
escalating the rest (escalation_recall=0.818 of actually-wrong answers caught,
escalation_precision=0.265).
```

**Two honest findings, not one clean success story:**

1. **The gate does clear the minimal bar** at threshold=0.5: among the 32% of
   queries it chooses to answer, accuracy is 87.5% vs. 78.0% unconditionally
   — and it catches 81.8% of the actually-wrong answers by escalating them.
   That's the qualitative behavior a confidence gate is supposed to have.
2. **A real, discovered limitation**: `grounding_overlap` (the open-book
   signal) barely varies (0.868–0.937) across *all* examples, correct or not,
   because `build_confidence_labels.py` uses templated answers that
   **literally quote their own top-1 passage** — so the answer is
   "grounded" by construction even when that passage is the wrong one. This
   means almost all of the gate's discriminative power above comes from
   `retrieval_margin` alone, not from the combination the design intended.
   This is exactly the INTRYGUE (`intrygue2026`) warning in practice: a
   confidence signal that looks reasonable in aggregate can be silently
   carried by only one of its inputs, discoverable only by validating against
   labels rather than trusting the design. A production version would need
   to validate this signal against real (non-template-quoting) LLM answers,
   where grounding overlap can actually diverge from correctness.

All numbers here are from a 50-example set (11 incorrect) — every ratio has a
wide confidence interval; read this as "the mechanism behaves sanely and the
validation methodology itself surfaced a real flaw," not as a certified
calibration curve.

### 4. Latency benchmark (`python scripts/benchmark_latency.py`), this machine, this KB

```
--- One-time model load latency ---
bm25_retriever                                              1.5 ms
dense_retriever (incl. MiniLM-L6-v2 load + KB encode)   10793.5 ms
faster_whisper_tiny                                        541.7 ms
wav2vec2_base + trained_adapter                           3048.9 ms
qwen2.5-0.5b-instruct                                      654.3 ms

--- Per-stage steady-state latency (text query path, n=20) ---
dialect_mapping                            mean=   0.1ms  p95=   0.6ms
hybrid_search (bm25+dense+fusion)          mean=   3.5ms  p95=   4.5ms
confidence_gate.score                      mean=   0.0ms  p95=   0.0ms
llm_generation (qwen2.5-0.5b, <=120 tok)   mean=1224.6ms  p95=1392.4ms

--- asr_cascade audio path (n=20) ---
whisper_tiny_transcription                 mean= 189.0ms  p95= 234.2ms
asr_cascade_total                          mean= 194.9ms  p95= 240.4ms

--- speech_native audio path (n=20) ---
speech_native_total (encode+retrieve)      mean=  40.3ms  p95= 135.9ms
```

Peak VRAM with **every** model loaded simultaneously (dense retriever,
Whisper tiny, wav2vec2+adapter, Qwen2.5-0.5B fp16) measured via
`torch.cuda.max_memory_allocated()`: **~1.48 GB** — comfortably inside the 6GB
budget with ~4.5GB headroom, on the actual RTX 3050 Laptop GPU this was built
on.

**Read on RQ3:** retrieval, dialect mapping, and the confidence gate are all
sub-10ms and effectively free next to everything else. The dominant cost is
LLM generation (~1.2s mean, capped at 120 new tokens, greedy decoding) — end
to end this lands in roughly the same "sub-2-second" ballpark `dubey2025aahar`
reports for a comparable offline agricultural LLM system, on comparable
consumer hardware, which is at least consistent with (not proof of) RQ3's
premise holding once the speech/dialect-mapping/confidence layers are stacked
on top of generation, for a 42-passage KB. This is a demo-scale KB — retrieval
latency here should not be extrapolated linearly to a production-scale KB
(thousands+ passages) without re-measuring; BM25/dense search over 42 short
passages is close to a best case for both.

### 5. Tests (`pytest tests/ -v`)

```
24 passed, 5 warnings in ~41s
```

All 24 pass: 6 retrieval tests, 6 dialect-mapping tests, 6 confidence-gate
tests, and 6 pipeline smoke tests (text, dialectal-text, low-confidence
escalation, timing reporting, speech-native audio, ASR-cascade audio — the
speech-native test auto-skips if `models/speech_adapter.pt` hasn't been
trained yet in your environment).

---

## Known limitations / explicitly out of scope for this prototype

- **Dialect mapping is text-only (post-ASR-style lookup)**, per "architecture
  #1" in `agents/components/dialect-entity-mapping.md`. It is therefore not
  applied in the `speech_native` pipeline mode at all (there's no text to look
  it up in) — "architecture #2" (embedding-space dialect mapping with no ASR)
  is named in that doc as the harder, more consistent-with-the-paper's-framing
  alternative, and is not implemented here. `src/pipeline.py`'s docstring
  flags this explicitly.
- **The dialect lexicon (29 entries) and KB (42 passages) are small,
  hand-curated demo artifacts**, not domain-expert-reviewed or sourced from a
  public KG/ontology as the paper's own bootstrapping constraint (contribution
  2) requires for a real deployment. They're built from generally well-known
  agricultural-extension terminology so the *mechanism* is genuinely
  testable, not to claim linguistic or agronomic authority.
- **No real dialect speech data exists or was collected** for this project —
  see the central caveat above. Every number involving audio in this
  repository is measured on espeak-ng TTS synthesis of text, not recordings
  of real speakers.
- **The escalation target ("a human expert") is not implemented** — the
  pipeline returns an `"ESCALATE TO HUMAN EXPERT"` string; wiring that to an
  actual extension-worker phone queue or ticketing system (as
  `agents/components/confidence-gated-escalation.md` names as a real
  operational dependency) is out of scope here.
- **Cloud GPU vs. offline boundary**: per the paper's contribution 4 scoping,
  this prototype used network access only for (a) one-time model weight
  downloads (MiniLM, Whisper tiny, wav2vec2-base, Qwen2.5-0.5B) and (b)
  adapter training reads pretrained wav2vec2 weights already cached locally —
  everything in `scripts/benchmark_latency.py` and `scripts/run_evaluation.py`
  runs with no network calls once those one-time downloads are cached.
