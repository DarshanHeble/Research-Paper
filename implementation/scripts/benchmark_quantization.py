#!/usr/bin/env python
"""Benchmarks the local LLM generation stage (Qwen2.5-0.5B-Instruct) under three
quantization modes -- fp16 (the baseline reported elsewhere in this repo and in
main.tex Table III), int8, and int4 (both via bitsandbytes) -- to test whether
the compactllm2026-style throughput figures cited in main.tex Section III-F/IV-F
actually hold once quantization is applied to THIS exact pipeline's generation
stage, on THIS exact hardware (RTX 3050 Laptop, 6GB VRAM). This closes the
"quantization of the local LLM" item main.tex's Conclusion explicitly names as
still-open future work -- this script provides the actual measurement.

Run with --mode {fp16,int8,int4,all} (default: all). Each mode is measured in
its OWN fresh subprocess (see main()), because loading three different-dtype
copies of the same model in one process would (a) not fit in 6GB VRAM
simultaneously and (b) make torch.cuda.max_memory_allocated() report a
compounded peak across modes rather than each mode's own honest peak. This is
the same reason benchmark_latency.py measures peak VRAM within a single
process for the FULL pipeline (all distinct models loaded together, which is
the real deployment shape) -- this script instead isolates one component
(generation) across three alternative configurations of itself, which needs
process isolation, not shared-process measurement.

Only the generation stage is re-measured here -- dialect mapping, retrieval,
and the confidence gate are unaffected by LLM quantization and are already
benchmarked in benchmark_latency.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_QUERIES_PATH = REPO_ROOT / "data" / "eval_queries.jsonl"
KB_PATH = REPO_ROOT / "data" / "kb.json"
N_SAMPLE = 20


def _run_one_mode(mode: str) -> dict:
    """Runs in the CURRENT process -- called only via the subprocess re-invocation
    below, so each mode gets a clean CUDA context and an honest peak-VRAM reading."""
    import torch

    from src.generation.llm_generator import LocalLLMGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This benchmark requires a CUDA device.")
    torch.cuda.reset_peak_memory_stats()

    quantization = None if mode == "fp16" else mode
    t0 = time.time()
    llm = LocalLLMGenerator(device=device, quantization=quantization)
    load_s = time.time() - t0

    queries = []
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    queries = queries[:N_SAMPLE]

    with open(KB_PATH, encoding="utf-8") as f:
        passages = json.load(f)
    sample_passage = passages[0]["text"]

    # one warmup call, excluded from steady-state stats (first CUDA kernel launch/
    # cuDNN autotune absorbs one-time JIT cost, same convention as benchmark_latency.py)
    llm.generate(queries[0]["text"], sample_passage)

    gen_times = []
    for q in queries:
        t0 = time.time()
        llm.generate(q["text"], sample_passage)
        gen_times.append(time.time() - t0)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    times_sorted = sorted(gen_times)
    p95_idx = min(len(times_sorted) - 1, int(0.95 * len(times_sorted)))

    return {
        "mode": mode,
        "load_s": load_s,
        "n": len(gen_times),
        "mean_ms": statistics.mean(gen_times) * 1000,
        "median_ms": statistics.median(gen_times) * 1000,
        "p95_ms": times_sorted[p95_idx] * 1000,
        "min_ms": min(gen_times) * 1000,
        "max_ms": max(gen_times) * 1000,
        "peak_vram_gb": peak_vram_gb,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fp16", "int8", "int4", "all"], default="all")
    parser.add_argument("--_subprocess-mode", dest="_subprocess_mode", default=None,
                         help=argparse.SUPPRESS)  # internal: set when re-invoked as a worker
    args = parser.parse_args()

    if args._subprocess_mode:
        result = _run_one_mode(args._subprocess_mode)
        print("SUBPROCESS_RESULT_JSON:" + json.dumps(result))
        return

    modes = ["fp16", "int8", "int4"] if args.mode == "all" else [args.mode]
    results = []
    for mode in modes:
        print(f"\n=== Benchmarking mode={mode} (fresh subprocess for a clean CUDA context) ===")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_subprocess-mode", mode],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"[mode={mode} FAILED, returncode={proc.returncode}]", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            results.append({"mode": mode, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr else "unknown error"})
            continue
        for line in proc.stdout.splitlines():
            if line.startswith("SUBPROCESS_RESULT_JSON:"):
                results.append(json.loads(line[len("SUBPROCESS_RESULT_JSON:"):]))

    print("\n=== Summary: LLM generation latency + peak VRAM by quantization mode ===")
    print(f"{'mode':6s} {'load_s':>8s} {'mean_ms':>9s} {'median_ms':>10s} {'p95_ms':>8s} {'peak_vram_gb':>13s}")
    for r in results:
        if "error" in r:
            print(f"{r['mode']:6s}  FAILED: {r['error']}")
            continue
        print(f"{r['mode']:6s} {r['load_s']:8.1f} {r['mean_ms']:9.1f} {r['median_ms']:10.1f} "
              f"{r['p95_ms']:8.1f} {r['peak_vram_gb']:13.2f}")

    out_path = REPO_ROOT / "data" / "quantization_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {out_path}")
    print("\nNote: peak_vram_gb here is for the LLM generation stage ALONE, in isolation (one model "
          "loaded per subprocess), not the full-pipeline peak reported in benchmark_latency.py (which "
          "loads the dense retriever, Whisper, wav2vec2+adapter, and the LLM together). To get a "
          "full-pipeline peak under a quantized LLM, the fp16 LLM in that script's load sequence "
          "would need to be swapped for a quantized one -- not done here, so don't conflate the two "
          "numbers.")


if __name__ == "__main__":
    main()
