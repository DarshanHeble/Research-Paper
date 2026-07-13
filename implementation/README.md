# Implementation — Speech-Native RAG for Agricultural Advisory (Prototype)

This directory is a **best-effort, genuinely runnable software prototype** of
the system proposed in `paper/main.tex` and specified in `agents/implementation.md`
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

This prototype has now been built and run end to end on two different
machines, both with the same GPU class, which is itself a small piece of
evidence for the "consumer-grade, GPU-constrained hardware" framing
(Section III-F) not being a fragile, one-machine result:

- **Original build**: Python 3.12.3, Ubuntu (Pop!_OS), 12 cores, 15GB RAM.
- **Reproduction run** (this session): Python 3.12.13, Arch Linux, same host
  class. Fresh venv, fresh model downloads, fresh training run, fresh
  evaluation — every number in the Results section below is from this
  reproduction run, not copy-pasted from the original build.
- GPU (both machines): NVIDIA RTX 3050 Laptop, 6GB VRAM — used as the actual
  latency-benchmark target, per the paper's own "consumer-grade,
  GPU-constrained hardware" framing (Section III-F). This is a real instance
  of that hardware class, not a stand-in for it.
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

This session's reproduction hit the same stalled-resumable-download pattern
again, this time on a *different* file: `huggingface_hub`'s chunked
downloader for `facebook/wav2vec2-base` (used by the speech-native adapter,
not the LLM) stalled indefinitely partway through a secondary blob after
~40 minutes with zero throughput on an otherwise-idle TCP connection, while
`pytorch_model.bin` itself (the file actually needed to load the model) had
already downloaded successfully in full. The fix was the same each time:
kill the stalled process, delete the `.incomplete` blob and its orphaned
snapshot directory from `~/.cache/huggingface/hub/`, and retry — the needed
files were already cached, so the retry completed in seconds. This is now
two independent occurrences of the same failure mode on two different
models and two different machines, which is reasonably strong evidence it's
a `huggingface_hub` resumable-download issue rather than a one-off fluke of
either specific file or host.

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
- `espeak-ng` itself was not preinstalled in either reference environment this
  project has been built in, and passwordless sudo wasn't available in
  either. Two no-root extraction methods, both leaving a self-contained tree
  under `tools/` (no source compiled, no system files touched):
  - Original build (Ubuntu): `apt-get download` (no root needed to download)
    the `.deb`, single binary extracted with `dpkg -x` into `tools/espeak-ng`,
    relying on `espeak-ng-data`/`libespeak-ng1` already present system-wide.
  - Reproduction run (Arch, this session): `pacman -Sp` (prints the mirror
    URL without installing) for `espeak-ng` and its two link-time deps
    (`pcaudiolib`, `libsonic`), fetched with `curl` and unpacked with
    `tar --zstd -x` into `tools/espeak-ng-dist/{bin,lib,espeak-ng-data}/` —
    fully self-contained, no system-wide espeak files needed at all, run with
    `LD_LIBRARY_PATH`/`ESPEAK_DATA_PATH` pointed at that tree.
  `src/tts_util.py` supports both layouts and picks whichever is present; see
  its docstring for the full story.

### A mid-session environment issue from the original build (Ubuntu), not seen in the reproduction

Partway through the *original* build, an unattended `nvidia-driver` package
upgrade on that machine updated the userspace driver library while the
already-loaded kernel module stayed on an older version, breaking
`nvidia-smi`/NVML for the rest of that session (no reboot or root access
available to fix it mid-session). `torch.cuda` compute itself kept working
throughout — verified with a real on-GPU matmul producing a correct result —
only the NVML-based monitoring calls were affected. `scripts/benchmark_latency.py`
used to print a hardcoded note about this on every run regardless of whether
it was still true; that was a leftover from the original session baked into
the script rather than a live check, and has been removed as part of this
session's reproduction (`nvidia-smi` works normally on the Arch reproduction
machine). The script now measures peak VRAM directly via
`torch.cuda.max_memory_allocated()` instead of relying on `nvidia-smi` at all
— see the latency results below.

---

## Results — real, measured, on this demo data (paste, not paraphrase)

**Provenance note:** the numbers below are from the Arch Linux reproduction
run performed in this session (fresh venv, fresh model downloads, fresh
adapter training, fresh evaluation) — not copy-pasted from the original
Ubuntu build. Where a number differs slightly from an earlier version of this
file, that's genuine run-to-run variance (GPU non-determinism in cuDNN
kernels, and — for baselines 5/6 below — `faster-whisper`/`wav2vec2` version
drift between the two environments), not a transcription error; both runs
tell the same qualitative story on every metric that matters for the paper's
RQs.

### 1. Retrieval baselines (`python scripts/run_evaluation.py`), 50 stratified eval queries

Full output in `data/evaluation_results.json`. Headline recall@1 by baseline
and stratum:

| Baseline | overall | standard/common | standard/rare | dialectal/common | dialectal/rare |
|---|---:|---:|---:|---:|---:|
| 1. keyword-only (BM25) | 0.500 | 0.917 | 1.000 | 0.077 | 0.077 |
| 2. dense-only | 0.400 | 0.833 | 0.833 | 0.000 | 0.000 |
| 3. hybrid, no dialect mapping | 0.460 | 0.917 | 1.000 | 0.000 | 0.000 |
| 4. hybrid + dialect mapping | **0.780** | 0.917 | 1.000 | **0.615** | **0.615** |
| 5. ASR cascade (Whisper tiny) + hybrid + dialect mapping | 0.400 | 0.583 | 1.000 | 0.077 | 0.000 |
| 6. speech-native (trained adapter), no dialect mapping | 0.020 | 0.000 | 0.000 | 0.000 | 0.077 |

**What this shows, honestly:**
- **Dialect mapping is doing essentially all of the work on dialectal
  queries** — every baseline without it scores ~0.0 recall@1 on
  dialectal/common and dialectal/rare; adding it (#4) jumps both to 0.615.
  This is a real, clean confirmation of RQ2/RQ4's premise on this demo set —
  and also a sign that 42 KB passages with fairly distinct vocabulary is an
  easy regime for a lexicon-lookup layer; it would need re-testing at a
  larger, messier KB scale before trusting the same 0.615 number.
- **ASR cascade is genuinely, badly broken by dialectal queries** — 0.077 and
  0.000 recall@1 on the two dialectal strata, and worse than raw hybrid
  retrieval even on standard-phrasing queries (0.583 vs. 0.917), because
  `faster-whisper` "tiny" mangles even the *English* TTS audio somewhat (see
  `data/asr_cascade_transcripts.json` for every transcript — e.g. query q001
  "My rice field has hopperburn with yellowing and drying at the base of the
  tillers..." was transcribed as "My life feels first hot women with
  yellowing and drying at the face of the t-run..."). Whisper "tiny" on
  robotic TTS audio is a harder condition than Whisper on natural human
  speech; this number should not be read as "Whisper is this bad in
  general," only that this specific tiny model + this specific synthetic
  audio pipeline is.
- **Speech-native retrieval's recall@1 on the held-out eval set is 0.020
  overall** (1/50, effectively chance), despite the adapter clearly learning
  *something* (see below) — a real, and important, negative result: the toy
  training signal does not generalize to the eval set's different phrasing
  style. This is discussed in depth below, not glossed over.

### 2. Speech-native adapter training (`python -m src.speech_retrieval.train_adapter`)

```
[1/5] 108 train examples, 18 val examples (auto-generated templates, disjoint
      from eval_queries.jsonl -- see train_adapter.py docstring)
...
[4/5] Training adapter for 100 epochs with in-batch InfoNCE contrastive loss...
    epoch   1  train_loss=3.5050  val_top1_passage_acc=0.000
    epoch  50  train_loss=0.2842  val_top1_passage_acc=0.333
    epoch  90  train_loss=0.1699  val_top1_passage_acc=0.444
    epoch 100  train_loss=0.0888  val_top1_passage_acc=0.500
[5/5] Best checkpoint: epoch 99, val_top1_passage_acc=0.611 (chance level = 1/42 = 0.024)
Total wall time: 20.1s
```

**The honest read:** on its OWN held-out validation split (18 examples, drawn
from the same small set of auto-generated English templates used for
training, e.g. *"What can I do about {entity} in {crop}?"*), the trained
adapter retrieves the correct passage top-1 **61.1% of the time — roughly 26x
better than the 2.4% chance level for a 42-way retrieval problem.** (The
original build's run landed at 55.6%/epoch 87 — same story, small
run-to-run variance from GPU non-determinism given a fixed random seed.)
That is a real, positive demonstration that the SpeechRAG/S2R-style mechanism
(frozen speech encoder → small trained adapter → frozen text-retriever
embedding space, no transcription) works end to end, reproducibly, across two
different machines.

But on `data/eval_queries.jsonl` — full natural-language symptom-description
sentences and romanized Hindi/Bengali queries, a genuinely different phrasing
distribution than the short template sentences it trained on — recall@1 drops
to 0.020 overall (see table above). **This is a real generalization gap, not
a bug**, and it's exactly the kind of result the paper predicts is
unresolved: a small synthetic/template training signal does not obviously
transfer to realistic out-of-distribution phrasing or (much more importantly)
real dialect speech, which is the actual open problem the paper names and
does not claim to have solved. Don't read the 61.1% number as "the adapter
works" without also reading the ~2% eval number right next to it.

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
bm25_retriever                                              1.4 ms
dense_retriever (incl. MiniLM-L6-v2 load + KB encode)    8611.1 ms
faster_whisper_tiny                                        453.3 ms
wav2vec2_base + trained_adapter                           2194.4 ms
qwen2.5-0.5b-instruct                                      695.0 ms

--- Per-stage steady-state latency (text query path, n=20) ---
dialect_mapping                            mean=   0.1ms  p95=   0.6ms
hybrid_search (bm25+dense+fusion)          mean=   3.8ms  p95=   4.3ms
confidence_gate.score                      mean=   0.0ms  p95=   0.1ms
llm_generation (qwen2.5-0.5b, <=120 tok)   mean=1282.5ms  p95=1476.4ms

--- asr_cascade audio path (n=20) ---
whisper_tiny_transcription                 mean= 212.8ms  p95= 288.3ms
asr_cascade_total                          mean= 218.5ms  p95= 293.3ms

--- speech_native audio path (n=20) ---
speech_native_total (encode+retrieve)      mean=  41.2ms  p95= 153.2ms

Peak VRAM with every model loaded simultaneously so far in this process: 1.54 GB
```

Peak VRAM (dense retriever, Whisper tiny, wav2vec2+adapter, Qwen2.5-0.5B
fp16, all loaded, plus 20 queries of actual inference through every stage)
measured via `torch.cuda.max_memory_allocated()`, now built into the script
itself rather than reported separately: **1.54 GB** — comfortably inside the
6GB budget with ~4.5GB headroom, on the actual RTX 3050 Laptop GPU this was
benchmarked on. (The original build's more limited "just after loading, no
generation yet" measurement was 1.48GB — consistent, small difference
explained by that run not having pushed a generation batch through yet.)

**Read on RQ3:** retrieval, dialect mapping, and the confidence gate are all
sub-10ms and effectively free next to everything else. The dominant cost is
LLM generation (~1.3s mean, capped at 120 new tokens, greedy decoding) — end
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
24 passed, 3 warnings in 19.04s
```

All 24 pass: 6 retrieval tests, 6 dialect-mapping tests, 6 confidence-gate
tests, and 6 pipeline smoke tests (text, dialectal-text, low-confidence
escalation, timing reporting, speech-native audio, ASR-cascade audio). In
this reproduction run `models/speech_adapter.pt` was trained first, so all 24
ran for real, including the speech-native smoke test (it auto-skips only if
no trained adapter is present).

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
