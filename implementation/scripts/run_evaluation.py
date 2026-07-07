#!/usr/bin/env python
"""Runs all retrieval baselines against data/eval_queries.jsonl and reports
recall@1/@3 and MRR, broken down by the 2x2 stratification (phrasing x rarity),
mirroring Section IV-C/D and the RQ1/RQ2/RQ4 comparisons.

Baselines:
  1. keyword_only     -- BM25 over raw (un-mapped) query text.
  2. dense_only        -- sentence-transformer dense retrieval over raw query text.
  3. hybrid            -- BM25+dense RRF fusion over raw query text (RQ4's premise).
  4. hybrid+dialect     -- hybrid retrieval over dialect-mapper-normalized query text (RQ2).
  5. asr_cascade        -- TTS-synthesized query audio -> faster-whisper transcription
                           -> hybrid+dialect retrieval on the transcript (RQ1 baseline).
  6. speech_native      -- TTS-synthesized query audio -> frozen wav2vec2 + trained
                           adapter -> direct embedding-space retrieval, no ASR, no
                           dialect mapping (documented limitation, see pipeline.py).

HONESTY NOTE: baselines 5 and 6 run on espeak-ng TTS audio of the eval queries
(including the romanized Hindi/Bengali dialectal ones), not real recorded dialect
speech -- see src/tts_util.py's module docstring for exactly why and what that
does and doesn't demonstrate. Numbers below are real, measured, on this demo
dataset -- they are not a claim about real-world dialect speech performance.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.dialect_mapping.mapper import DialectMapper  # noqa: E402
from src.tts_util import synthesize  # noqa: E402

KB_PATH = REPO_ROOT / "data" / "kb.json"
LEXICON_PATH = REPO_ROOT / "data" / "dialect_lexicon.json"
EVAL_QUERIES_PATH = REPO_ROOT / "data" / "eval_queries.jsonl"
EVAL_AUDIO_DIR = REPO_ROOT / "data" / "synthetic_audio" / "eval"
ADAPTER_PATH = REPO_ROOT / "models" / "speech_adapter.pt"

TOP_K = 5


def load_eval_queries() -> list[dict]:
    queries = []
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def rank_of_first_gold(retrieved_ids: list[str], gold_ids: list[str]) -> int | None:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return rank
    return None


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> int:
    return int(any(doc_id in gold_ids for doc_id in retrieved_ids[:k]))


class MetricAccumulator:
    def __init__(self):
        self.rows = []  # (stratum_key, recall@1, recall@3, reciprocal_rank)

    def add(self, phrasing: str, rarity: str, retrieved_ids: list[str], gold_ids: list[str]):
        rank = rank_of_first_gold(retrieved_ids, gold_ids)
        rr = 1.0 / rank if rank else 0.0
        r1 = recall_at_k(retrieved_ids, gold_ids, 1)
        r3 = recall_at_k(retrieved_ids, gold_ids, 3)
        self.rows.append((phrasing, rarity, r1, r3, rr))

    def report(self) -> dict:
        def summarize(rows):
            n = len(rows)
            if n == 0:
                return {"n": 0, "recall@1": None, "recall@3": None, "mrr": None}
            return {
                "n": n,
                "recall@1": sum(r[2] for r in rows) / n,
                "recall@3": sum(r[3] for r in rows) / n,
                "mrr": sum(r[4] for r in rows) / n,
            }

        out = {"overall": summarize(self.rows)}
        for phrasing in ("standard", "dialectal"):
            for rarity in ("common", "rare"):
                subset = [r for r in self.rows if r[0] == phrasing and r[1] == rarity]
                out[f"{phrasing}/{rarity}"] = summarize(subset)
        for phrasing in ("standard", "dialectal"):
            subset = [r for r in self.rows if r[0] == phrasing]
            out[f"phrasing={phrasing}"] = summarize(subset)
        for rarity in ("common", "rare"):
            subset = [r for r in self.rows if r[1] == rarity]
            out[f"rarity={rarity}"] = summarize(subset)
        return out


def print_report(name: str, report: dict):
    print(f"\n=== {name} ===")
    order = ["overall", "phrasing=standard", "phrasing=dialectal", "rarity=common", "rarity=rare",
             "standard/common", "standard/rare", "dialectal/common", "dialectal/rare"]
    print(f"{'stratum':22s} {'n':>4s} {'recall@1':>9s} {'recall@3':>9s} {'mrr':>7s}")
    for key in order:
        m = report[key]
        if m["n"] == 0:
            continue
        print(f"{key:22s} {m['n']:4d} {m['recall@1']:9.3f} {m['recall@3']:9.3f} {m['mrr']:7.3f}")


def main():
    print("Loading KB, retrievers, dialect mapper...")
    bm25 = BM25Retriever.from_kb_file(KB_PATH)
    dense = DenseRetriever.from_kb_file(KB_PATH)
    hybrid = HybridRetriever(bm25, dense)
    mapper = DialectMapper.from_lexicon_file(LEXICON_PATH)
    queries = load_eval_queries()
    print(f"Loaded {len(queries)} eval queries.")

    results = {}

    # ---- 1/2/3: keyword-only, dense-only, hybrid (raw text, no dialect mapping) ----
    for name, retrieve_fn in [
        ("1_keyword_only_bm25", lambda q: [h.doc_id for h in bm25.search(q, top_k=TOP_K)]),
        ("2_dense_only", lambda q: [h.doc_id for h in dense.search(q, top_k=TOP_K)]),
        ("3_hybrid_no_dialect_mapping", lambda q: [h.doc_id for h in hybrid.search(q, top_k=TOP_K)]),
    ]:
        acc = MetricAccumulator()
        t0 = time.time()
        for q in queries:
            retrieved = retrieve_fn(q["text"])
            acc.add(q["phrasing"], q["rarity"], retrieved, q["gold_ids"])
        report = acc.report()
        report["_wall_seconds"] = time.time() - t0
        results[name] = report
        print_report(name, report)

    # ---- 4: hybrid + dialect mapping ----
    acc = MetricAccumulator()
    t0 = time.time()
    for q in queries:
        normalized = mapper.map_query(q["text"]).normalized_query
        retrieved = [h.doc_id for h in hybrid.search(normalized, top_k=TOP_K)]
        acc.add(q["phrasing"], q["rarity"], retrieved, q["gold_ids"])
    report = acc.report()
    report["_wall_seconds"] = time.time() - t0
    results["4_hybrid_plus_dialect_mapping"] = report
    print_report("4_hybrid_plus_dialect_mapping", report)

    # ---- synthesize eval-query audio (shared by baselines 5 and 6) ----
    print("\nSynthesizing eval-query audio with espeak-ng (English voice; see src/tts_util.py "
          "for the honesty note on romanized-dialect TTS)...")
    EVAL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for q in queries:
        wav_path = EVAL_AUDIO_DIR / f"{q['id']}.wav"
        if not wav_path.exists():
            synthesize(q["text"], wav_path, voice="en")
        q["_wav_path"] = wav_path

    # ---- 5: ASR cascade (Whisper tiny) + hybrid + dialect mapping ----
    print("Loading faster-whisper (tiny)...")
    from src.asr.whisper_cascade import WhisperASRCascade

    asr_cascade = WhisperASRCascade(hybrid, model_size="tiny")
    acc = MetricAccumulator()
    t0 = time.time()
    transcripts_log = []
    for q in queries:
        transcript, _t = asr_cascade.transcribe(q["_wav_path"])
        normalized = mapper.map_query(transcript).normalized_query
        retrieved = [h.doc_id for h in hybrid.search(normalized, top_k=TOP_K)]
        acc.add(q["phrasing"], q["rarity"], retrieved, q["gold_ids"])
        transcripts_log.append({"id": q["id"], "original": q["text"], "transcript": transcript})
    report = acc.report()
    report["_wall_seconds"] = time.time() - t0
    results["5_asr_cascade_whisper_tiny"] = report
    print_report("5_asr_cascade_whisper_tiny", report)

    # ---- 6: speech-native retrieval (frozen wav2vec2 + trained adapter) ----
    if ADAPTER_PATH.exists():
        print("Loading speech-native retriever (frozen wav2vec2 + trained adapter)...")
        import librosa

        from src.speech_retrieval.speech_retriever import SpeechNativeRetriever

        speech_retriever = SpeechNativeRetriever(dense, ADAPTER_PATH)
        acc = MetricAccumulator()
        t0 = time.time()
        for q in queries:
            wav, _sr = librosa.load(str(q["_wav_path"]), sr=16000, mono=True)
            retrieved = [h.doc_id for h in speech_retriever.search(wav.astype("float32"), top_k=TOP_K)]
            acc.add(q["phrasing"], q["rarity"], retrieved, q["gold_ids"])
        report = acc.report()
        report["_wall_seconds"] = time.time() - t0
        results["6_speech_native_adapter"] = report
        print_report("6_speech_native_adapter", report)
    else:
        print(f"\n[SKIPPED] 6_speech_native_adapter -- no trained adapter at {ADAPTER_PATH}. "
              "Run `python -m src.speech_retrieval.train_adapter` first.")
        results["6_speech_native_adapter"] = None

    out_path = REPO_ROOT / "data" / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {out_path}")

    transcripts_path = REPO_ROOT / "data" / "asr_cascade_transcripts.json"
    with open(transcripts_path, "w", encoding="utf-8") as f:
        json.dump(transcripts_log, f, indent=2)
    print(f"ASR transcripts (for manual inspection of the mis-transcription failure mode) written to {transcripts_path}")


if __name__ == "__main__":
    main()
