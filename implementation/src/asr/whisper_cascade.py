"""ASR-cascaded retrieval baseline -- Section IV-C baseline #1 / RQ1 comparison.

query audio -> faster-whisper transcription -> (optional dialect mapping) -> hybrid
text retrieval.

This is the baseline that speech-native retrieval (speech_retrieval/) is meant to
be compared against. It embodies exactly the failure mode Section III-B/RQ1 is
about: if Whisper mis-transcribes a rare, domain-critical term (a pesticide/pest
name it has never seen), that error propagates into a bad retrieval query with no
opportunity for the retriever to recover, because by the time retrieval runs the
original acoustic signal is gone.

Model: faster-whisper "tiny" (or "base") -- chosen per the environment brief to
fit comfortably in 6GB VRAM alongside the other loaded models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedPassage


@dataclass
class ASRCascadeResult:
    transcript: str
    transcription_seconds: float
    retrieved: list[RetrievedPassage]


class WhisperASRCascade:
    def __init__(self, hybrid_retriever: HybridRetriever, model_size: str = "tiny",
                 device: str | None = None, compute_type: str | None = None):
        self.hybrid = hybrid_retriever
        self.device = device or _pick_device()
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path: str | Path) -> tuple[str, float]:
        import time

        t0 = time.time()
        segments, _info = self.model.transcribe(str(audio_path), language="en", beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip(), time.time() - t0

    def answer(self, audio_path: str | Path, top_k: int = 5) -> ASRCascadeResult:
        transcript, t_sec = self.transcribe(audio_path)
        retrieved = self.hybrid.search(transcript, top_k=top_k)
        return ASRCascadeResult(transcript=transcript, transcription_seconds=t_sec, retrieved=retrieved)


def _pick_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    import sys

    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever

    kb_path = Path(__file__).resolve().parents[2] / "data" / "kb.json"
    bm25 = BM25Retriever.from_kb_file(kb_path)
    dense = DenseRetriever.from_kb_file(kb_path)
    hybrid = HybridRetriever(bm25, dense)
    cascade = WhisperASRCascade(hybrid, model_size="tiny")

    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    if audio_path is None:
        print("Usage: python -m src.asr.whisper_cascade <audio_path.wav>")
        sys.exit(1)
    result = cascade.answer(audio_path)
    print(f"transcript ({result.transcription_seconds:.2f}s): {result.transcript!r}")
    for r in result.retrieved[:3]:
        print(f"  {r.score:6.4f}  {r.doc_id}  {r.passage['title']}")
