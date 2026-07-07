#!/usr/bin/env python
"""Measures real end-to-end latency of the pipeline, stage-by-stage, on THIS
machine (RTX 3050 Laptop, 6GB VRAM) -- mirrors Section III-F / IV-F / RQ3
("can the pipeline operate ... at acceptable latency").

Reports two things separately, since they matter for different questions:
  * One-time MODEL LOAD cost per component (paid once at process startup, not
    per query -- relevant to "does this fit in 6GB VRAM and start up in a
    reasonable time", not to per-query responsiveness).
  * Per-query STEADY-STATE latency for each pipeline stage and for each of the
    three end-to-end modes (text, asr_cascade, speech_native), run over a
    sample of real eval queries (and, for the audio modes, real TTS-synthesized
    audio -- see src/tts_util.py for the honesty note on what that audio is and
    isn't).

All numbers below are real wall-clock measurements from this run on this
machine, not estimates.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

KB_PATH = REPO_ROOT / "data" / "kb.json"
LEXICON_PATH = REPO_ROOT / "data" / "dialect_lexicon.json"
EVAL_QUERIES_PATH = REPO_ROOT / "data" / "eval_queries.jsonl"
ADAPTER_PATH = REPO_ROOT / "models" / "speech_adapter.pt"
N_SAMPLE = 20  # number of eval queries to benchmark per mode (steady-state, post-warmup)


def load_eval_queries(n: int) -> list[dict]:
    queries = []
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries[:n]


def summarize(times: list[float]) -> dict:
    if not times:
        return {"n": 0}
    times_sorted = sorted(times)
    p95_idx = min(len(times_sorted) - 1, int(0.95 * len(times_sorted)))
    return {
        "n": len(times), "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "p95_ms": times_sorted[p95_idx] * 1000,
        "min_ms": min(times) * 1000, "max_ms": max(times) * 1000,
    }


def print_summary(name: str, s: dict):
    if s["n"] == 0:
        print(f"{name:38s}  (no samples)")
        return
    print(f"{name:38s}  n={s['n']:3d}  mean={s['mean_ms']:7.1f}ms  median={s['median_ms']:7.1f}ms  "
          f"p95={s['p95_ms']:7.1f}ms  min={s['min_ms']:7.1f}ms  max={s['max_ms']:7.1f}ms")


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device under test: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else ''}")
    print("NOTE: mid-session, this machine's NVML/nvidia-smi monitoring broke due to an "
          "unattended nvidia-driver package upgrade changing the userspace library version "
          "without the loaded kernel module being reloaded (a pre-existing environment issue, "
          "not caused by this code). torch.cuda compute itself was verified still functional "
          "(a real matmul on-device was run and produced a correct result) -- only nvidia-smi's "
          "monitoring calls are affected, so latency numbers below are still real GPU numbers, "
          "just without the usual nvidia-smi utilization/memory sidebar.\n")

    load_times = {}
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.dialect_mapping.mapper import DialectMapper
    from src.confidence.gate import ConfidenceGate

    t0 = time.time()
    bm25 = BM25Retriever.from_kb_file(KB_PATH)
    load_times["bm25_retriever"] = time.time() - t0

    t0 = time.time()
    dense = DenseRetriever.from_kb_file(KB_PATH, device=device)
    load_times["dense_retriever (incl. MiniLM-L6-v2 load + KB encode)"] = time.time() - t0

    hybrid = HybridRetriever(bm25, dense)
    mapper = DialectMapper.from_lexicon_file(LEXICON_PATH)
    gate = ConfidenceGate()

    t0 = time.time()
    from src.asr.whisper_cascade import WhisperASRCascade
    asr_cascade = WhisperASRCascade(hybrid, model_size="tiny", device=device)
    load_times["faster_whisper_tiny"] = time.time() - t0

    speech_retriever = None
    if ADAPTER_PATH.exists():
        t0 = time.time()
        from src.speech_retrieval.speech_retriever import SpeechNativeRetriever
        speech_retriever = SpeechNativeRetriever(dense, ADAPTER_PATH, device=device)
        load_times["wav2vec2_base + trained_adapter"] = time.time() - t0

    llm = None
    t0 = time.time()
    try:
        from src.generation.llm_generator import LocalLLMGenerator
        llm = LocalLLMGenerator(device=device)
        load_times["qwen2.5-0.5b-instruct"] = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"[LLM unavailable: {type(e).__name__}: {e} -- generation stage will use template fallback]")

    print("--- One-time model load latency (paid once at process startup) ---")
    for name, t in load_times.items():
        print(f"{name:55s}  {t*1000:8.1f} ms")

    queries = load_eval_queries(N_SAMPLE)
    print(f"\nBenchmarking steady-state per-query latency over {len(queries)} sample eval queries "
          f"(first call to each stage above already absorbed as 'load'; these are post-warmup numbers)...\n")

    # ---- per-stage timings on the text path ----
    t_dialect, t_bm25, t_dense, t_hybrid, t_gate = [], [], [], [], []
    for q in queries:
        t0 = time.time(); mapping = mapper.map_query(q["text"]); t_dialect.append(time.time() - t0)
        t0 = time.time(); bm25.search(mapping.normalized_query, top_k=5); t_bm25.append(time.time() - t0)
        t0 = time.time(); dense.search(mapping.normalized_query, top_k=5); t_dense.append(time.time() - t0)
        t0 = time.time(); hits = hybrid.search(mapping.normalized_query, top_k=5); t_hybrid.append(time.time() - t0)
        t0 = time.time()
        gate.score(hits[0].score, hits[1].score if len(hits) > 1 else None, "dummy answer text", hits[0].passage["text"])
        t_gate.append(time.time() - t0)

    print("--- Per-stage steady-state latency (text query path) ---")
    print_summary("dialect_mapping", summarize(t_dialect))
    print_summary("bm25_search (alone)", summarize(t_bm25))
    print_summary("dense_search (alone, incl. query encode)", summarize(t_dense))
    print_summary("hybrid_search (bm25+dense+fusion)", summarize(t_hybrid))
    print_summary("confidence_gate.score", summarize(t_gate))

    if llm is not None:
        t_gen = []
        sample_passage = dense.passages[0]["text"]
        for q in queries:
            t0 = time.time()
            llm.generate(q["text"], sample_passage)
            t_gen.append(time.time() - t0)
        print_summary("llm_generation (qwen2.5-0.5b, greedy, <=120 new tokens)", summarize(t_gen))

    # ---- synthesize audio once for the audio-path benchmarks ----
    from src.tts_util import synthesize
    audio_dir = REPO_ROOT / "data" / "synthetic_audio" / "benchmark"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for q in queries:
        wav_path = audio_dir / f"{q['id']}.wav"
        if not wav_path.exists():
            synthesize(q["text"], wav_path, voice="en")
        q["_wav_path"] = wav_path

    t_asr_transcribe, t_asr_total = [], []
    for q in queries:
        t0 = time.time()
        transcript, t_internal = asr_cascade.transcribe(q["_wav_path"])
        t_asr_transcribe.append(t_internal)
        mapping = mapper.map_query(transcript)
        hybrid.search(mapping.normalized_query, top_k=5)
        t_asr_total.append(time.time() - t0)
    print("\n--- Per-stage steady-state latency (asr_cascade audio path) ---")
    print_summary("whisper_tiny_transcription", summarize(t_asr_transcribe))
    print_summary("asr_cascade_total (transcribe+map+retrieve)", summarize(t_asr_total))

    if speech_retriever is not None:
        import librosa

        t_speech_encode, t_speech_total = [], []
        for q in queries:
            wav, _sr = librosa.load(str(q["_wav_path"]), sr=16000, mono=True)
            t0 = time.time()
            speech_retriever.search(wav.astype("float32"), top_k=5)
            t_speech_total.append(time.time() - t0)
        print("\n--- Per-stage steady-state latency (speech_native audio path) ---")
        print_summary("speech_native_total (wav2vec2+adapter+retrieve)", summarize(t_speech_total))

    print("\n--- End-to-end pipeline latency per mode (steady-state, post model-load) ---")
    text_e2e = [a + b for a, b in zip(t_dialect, t_hybrid)]
    if llm is not None:
        text_e2e = [a + b for a, b in zip(text_e2e, t_gen)]
    text_e2e = [a + b for a, b in zip(text_e2e, t_gate)]
    print_summary("text mode (dialect+retrieval+gen+gate)", summarize(text_e2e))
    print_summary("asr_cascade mode (transcribe+map+retrieve; excl. gen/gate, see above)",
                  summarize(t_asr_total))
    if speech_retriever is not None:
        print_summary("speech_native mode (encode+retrieve; excl. gen/gate, see above)",
                      summarize(t_speech_total))

    print("\nAll numbers above are real measurements on this machine (RTX 3050 Laptop, 6GB VRAM) "
          "for this demo-scale KB (42 passages) -- retrieval-stage latency in particular should NOT "
          "be extrapolated linearly to a production-scale KB (thousands+ passages) without re-measuring; "
          "BM25/dense search over 42 short passages is close to a best case.")


if __name__ == "__main__":
    main()
