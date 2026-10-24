"""Speech-native adapter: multi-seed run + embedding-space dialect-mapping ablation.

Trains the adapter (src/speech_retrieval/train_adapter.py) for each seed in two
modes -- dialect sentences distilled to their lexicon-NORMALIZED text embedding
("norm", the default recipe) vs. their RAW text embedding ("raw", ablation) --
and evaluates each on the held-out eval set's synthesized audio, same as
condition 6 of scripts/run_evaluation.py. Writes data/speech_native_ablation.json.

Run scripts/run_evaluation.py first so the eval audio exists. Same TTS honesty
caveat as everywhere else: synthetic English-voice audio, not dialect speech.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import librosa  # noqa: E402

from scripts.run_evaluation import (  # noqa: E402
    EVAL_AUDIO_DIR, KB_PATH, TOP_K, MetricAccumulator, load_eval_queries, print_report,
)
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.speech_retrieval import train_adapter  # noqa: E402
from src.speech_retrieval.speech_retriever import SpeechNativeRetriever  # noqa: E402

SEEDS = [42, 1, 2]
OUT_PATH = REPO_ROOT / "data" / "speech_native_ablation.json"


def main():
    queries = load_eval_queries()
    wavs = {q["id"]: librosa.load(str(EVAL_AUDIO_DIR / f"{q['id']}.wav"), sr=16000, mono=True)[0] for q in queries}
    dense = DenseRetriever.from_kb_file(KB_PATH)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("norm", "raw"):
            for seed in SEEDS:
                ckpt = Path(tmp) / f"{mode}_{seed}.pt"
                train_adapter.main(seed=seed, normalize_dialect=(mode == "norm"), out_path=ckpt)
                retriever = SpeechNativeRetriever(dense, ckpt)
                acc = MetricAccumulator()
                for q in queries:
                    ids = [h.doc_id for h in retriever.search(wavs[q["id"]], top_k=TOP_K)]
                    acc.add(q["phrasing"], q["rarity"], ids, q["gold_ids"])
                name = f"{mode}_seed{seed}"
                results[name] = acc.report()
                print_report(name, results[name])

    # Per-stratum mean over seeds, per mode.
    for mode in ("norm", "raw"):
        runs = [results[f"{mode}_seed{s}"] for s in SEEDS]
        results[f"{mode}_mean"] = {
            stratum: {m: sum(r[stratum][m] for r in runs) / len(runs) for m in ("recall@1", "recall@3", "mrr")}
            for stratum in runs[0]
        }
        print(f"\n{mode}_mean: " + ", ".join(f"{k}={v['recall@1']:.3f}" for k, v in results[f"{mode}_mean"].items()))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
